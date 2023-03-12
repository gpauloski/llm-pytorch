from __future__ import annotations

import dataclasses
import logging
import pprint
import sys
from collections.abc import Sequence

import torch.distributed as dist

from llm.config import flattened_config
from llm.datasets.bert import Batch
from llm.datasets.sharded import ResumableSequentialSampler
from llm.engine.initialize import initialize as engine_initialize
from llm.environment import log_environment
from llm.initialize import get_default_parser
from llm.initialize import initialize_from_args
from llm.loss import BertPretrainingCriterion
from llm.models import bert
from llm.optimizers import get_optimizer
from llm.schedulers import LinearWarmupLR
from llm.timer import Timer
from llm.trainers.bert.utils import checkpoint
from llm.trainers.bert.utils import get_dataloader
from llm.trainers.bert.utils import get_dataset
from llm.trainers.bert.utils import get_optimizer_grouped_parameters
from llm.trainers.bert.utils import load_state
from llm.trainers.bert.utils import parse_config
from llm.utils import create_summary_writer
from llm.utils import log_step

logger = logging.getLogger('llm.trainers.bert')

LOG_FMT = (
    'phase: {phase} | epoch: {epoch} | step: {step} | loss: {loss:.3f} | '
    'lr: {lr:.2e} | samples/s: {samples_per_second:.2f} | time (s): {time:.3f}'
)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = get_default_parser()
    args = parser.parse_args(argv)
    _config = initialize_from_args(args)
    log_environment()
    config = parse_config(_config)

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
        extra={'ranks': [0]},
    )
    logger.info(
        f'Config:\n{pprint.pformat(dataclasses.asdict(config))}',
        extra={'ranks': [0]},
    )

    model = bert.from_config(
        config.BERT_CONFIG,
        config.GRADIENT_CHECKPOINTING,
    )
    grouped_params = get_optimizer_grouped_parameters(model)
    optimizer = get_optimizer(config.OPTIMIZER, grouped_params, config.LR)
    criterion = BertPretrainingCriterion(config.BERT_CONFIG['vocab_size'])
    scheduler = LinearWarmupLR(
        optimizer,
        total_steps=config.STEPS,
        warmup_steps=config.WARMUP_STEPS,
    )

    model, optimizer, criterion, scheduler = engine_initialize(
        model,
        optimizer,
        criterion=criterion,
        scheduler=scheduler,
        accumulation_steps=config.ACCUMULATION_STEPS,
        dtype=config.DTYPE,
        max_norm=config.CLIP_GRAD_NORM,
    )

    global_step, epoch, sampler_index = load_state(
        config,
        model,
        optimizer,
        scheduler,
    )
    writer = create_summary_writer(
        config.TENSORBOARD_DIR,
        flattened_config(dataclasses.asdict(config)),
        ['train/loss', 'train/lr', 'train/epoch', 'train/samples_per_second'],
        purge_step=global_step,
    )
    logger.info(
        f'Writing TensorBoard logs to {config.TENSORBOARD_DIR}',
        extra={'ranks': [0]},
    )

    dataset = get_dataset(config.DATA_DIR)
    logger.info(
        f'Dataset size on rank 0: {len(dataset)}',
        extra={'ranks': [0]},
    )
    sampler = ResumableSequentialSampler(dataset, start_index=sampler_index)

    micro_step = 0
    global_step_timer = Timer(synchronize=True)
    step_loss = 0.0

    model.train()

    while global_step < config.STEPS:
        dataloader = get_dataloader(dataset, sampler, config.BATCH_SIZE)
        logger.info(
            f'Initialized new dataloader {len(dataloader)} batches',
            extra={'ranks': [0]},
        )

        for batch_index, batch in enumerate(dataloader):
            micro_step += 1

            batch = Batch(*[t.cuda() for t in batch])
            optimizer.zero_grad()
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                token_type_ids=batch.token_type_ids,
            )
            loss = criterion(
                prediction_scores=output.prediction_logits,
                masked_lm_labels=batch.masked_labels,
                seq_relationship_score=output.seq_relationship_logits,
                next_sentence_labels=batch.next_sentence_labels,
            )
            optimizer.backward(loss)
            optimizer.step()

            step_loss += loss.float().item()

            if micro_step % config.ACCUMULATION_STEPS == 0:
                global_step += 1
                step_time = global_step_timer.lap()

                samples = config.GLOBAL_BATCH_SIZE * dist.get_world_size()
                log_step(
                    logger,
                    step=global_step,
                    phase=config.PHASE,
                    epoch=epoch,
                    loss=step_loss / config.ACCUMULATION_STEPS,
                    samples_per_second=samples / step_time,
                    time=step_time,
                    lr=scheduler.get_last_lr()[0],
                    fmt_str=LOG_FMT,
                    writer=writer,
                    tensorboard_prefix='train',
                    skip_tensorboard=['phase', 'time'],
                )

                if (
                    global_step % config.CHECKPOINT_STEPS == 0
                    or global_step >= config.STEPS
                ):
                    checkpoint(
                        config,
                        global_step=global_step,
                        epoch=epoch,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        sampler_index=(batch_index + 1) * config.BATCH_SIZE,
                    )

                step_loss = 0.0

            # Wait to step scheduler until after we have logged current LR
            scheduler.step()

            if global_step >= config.STEPS:
                # Break out of dataset loop, then while loop will also break
                break

        epoch += 1

        # Reset sampler for next epoch
        sampler = ResumableSequentialSampler(dataset)

    writer.close()

    step_time = global_step_timer.get_history_mean()
    # Stop timer after getting avg step time because it will add a history
    # entry from between end of last step and now
    global_step_timer.stop()
    total_time = global_step_timer.get_history_sum()

    logger.info(
        'Training complete ('
        f'total training time (s): {total_time:.3f}, '
        f'avg step time (s): {step_time:.3f})',
        extra={'ranks': [0]},
    )
    return 0
