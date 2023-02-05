from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from transformers import BertForPreTraining

from llm.checkpoint import load_checkpoint
from llm.checkpoint import save_checkpoint
from llm.config import Config
from llm.config import flattened_config
from llm.datasets.nvidia import Batch
from llm.datasets.nvidia import sharded_dataset
from llm.engine.initialize import initialize as engine_initialize
from llm.initialize import get_default_parser
from llm.initialize import initialize_from_args
from llm.loss import BertPretrainingCriterion
from llm.models import bert
from llm.optimizers import get_optimizer
from llm.schedulers import LinearWarmupLR
from llm.timer import Timer
from llm.utils import create_summary_writer
from llm.utils import log_step

logger = logging.getLogger(__name__)

LOG_FMT = (
    'phase: {phase} | epoch: {epoch} | step: {step} | '
    'loss: {loss:.3f} | lr: {lr:.2e} | time (s): {time:.3f}'
)


def checkpoint(
    config: Config,
    global_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:  # pragma: no cover
    if dist.get_rank() == 0:
        save_checkpoint(
            checkpoint_dir=config.CHECKPOINT_DIR,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            phase=config.PHASE,
        )
    logger.info(
        f'Saved checkpoint at global step {global_step}',
        extra={'ranks': [0]},
    )


def load_state(
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:  # pragma: no cover
    global_step = 0
    epoch = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = load_checkpoint(config.CHECKPOINT_DIR, map_location='cpu')
    if checkpoint is not None:
        logger.info(
            f'Loaded checkpoint from {checkpoint.filepath}',
            extra={'ranks': [0]},
        )
        model.load_state_dict(checkpoint.model_state_dict)

        if checkpoint.kwargs['phase'] == config.PHASE:
            logger.info(
                'Checkpoint from current phase. Loading optimizer state',
                extra={'ranks': [0]},
            )
            if checkpoint.optimizer_state_dict is not None:
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                device = (
                    f'cuda:{torch.cuda.current_device()}'
                    if torch.cuda.is_available()
                    else 'cpu'
                )
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if checkpoint.scheduler_state_dict is not None:
                scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            global_step = checkpoint.global_step
            epoch = checkpoint.kwargs['epoch']
        else:
            logger.info(
                'Checkpoint from new phase. Resetting optimizer state',
                extra={'ranks': [0]},
            )
    else:
        logger.info(
            'No checkpoint found to resume from.',
            extra={'ranks': [0]},
        )

    return global_step, epoch


def get_optimizer_grouped_parameters(
    model: BertForPreTraining,
) -> list[dict[str, Any]]:  # pragma: no cover
    # configure the weight decay for bert models
    params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    params_decay = [p for n, p in params if not any(v in n for v in no_decay)]
    params_no_decay = [p for n, p in params if any(v in n for v in no_decay)]

    return [
        {'params': params_decay, 'weight_decay': 0.01},
        {'params': params_no_decay, 'weight_decay': 0.0},
    ]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = get_default_parser()
    args = parser.parse_args(argv)
    config = initialize_from_args(args)

    config.accumulation_steps = config.GLOBAL_BATCH_SIZE / (
        dist.get_world_size() * config.BATCH_SIZE
    )
    if config.accumulation_steps != int(config.accumulation_steps):
        raise ValueError(
            'Global batch size must be divisible by the product of the '
            'world size and batch size.',
        )

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
        extra={'ranks': [0]},
    )

    model = bert.from_config(
        config.BERT_CONFIG,
        config.get('GRADIENT_CHECKPOINTING', False),
    )
    grouped_params = get_optimizer_grouped_parameters(model)
    optimizer = get_optimizer(config.OPTIMIZER, grouped_params, config.LR)
    criterion = BertPretrainingCriterion(config.BERT_CONFIG['vocab_size'])
    scheduler = LinearWarmupLR(
        optimizer,
        total_steps=config.STEPS,
        warmup_steps=config.WARMUP_STEPS,
    )

    global_step, epoch = load_state(config, model, optimizer, scheduler)

    writer = create_summary_writer(
        config.TENSORBOARD_DIR,
        flattened_config(config),
        ['train/loss', 'train/lr', 'train/epoch'],
        purge_step=global_step,
    )

    engine, optimizer, criterion, scheduler = engine_initialize(
        model,
        optimizer,
        criterion=criterion,
        scheduler=scheduler,
        accumulation_steps=config.accumulation_steps,
        dtype=config.get('dtype', None),
        max_norm=config.get('max_norm_clip', None),
    )

    micro_step = 0
    global_step_timer = Timer()
    step_loss = 0.0

    model.train()

    while global_step < config.STEPS:
        dataset = sharded_dataset(
            config.DATA_DIR,
            config.BATCH_SIZE,
            seed=config.seed,
        )
        for batch in dataset:
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

            if micro_step % config.accumulation_steps == 0:
                global_step += 1

                log_step(
                    logger,
                    step=global_step,
                    phase=config.PHASE,
                    epoch=epoch,
                    loss=step_loss / config.accumulation_steps,
                    time=global_step_timer.lap(),
                    lr=scheduler.get_last_lr()[0],
                    fmt_str=LOG_FMT,
                    writer=writer,
                    tensorboard_prefix='train',
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
                    )

                step_loss = 0.0

            # Wait to step scheduler until after we have logged current LR
            scheduler.step()

            if global_step >= config.STEPS:
                # Break out of dataset loop, then while loop will also break
                break

        epoch += 1

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


if __name__ == '__main__':
    raise SystemExit(main())
