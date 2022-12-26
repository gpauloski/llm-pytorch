from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any
from typing import Literal

import colossalai
import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.nn.lr_scheduler import PolynomialWarmupLR
from colossalai.nn.optimizer import FusedAdam
from colossalai.nn.optimizer import FusedLAMB
from transformers import BertForPreTraining

from llm.datasets.nvidia import Batch
from llm.datasets.nvidia import sharded_dataset
from llm.loss import BertPretrainingCriterion
from llm.models import bert


def log_step(
    logger: colossalai.logger.DistributedLogger,
    *,
    epoch: int,
    global_step: int,
    step_loss: float,
    step_time: float,
    lr: float,
) -> None:  # pragma: no cover
    logger.info(
        f'epoch: {epoch} | step: {global_step} | loss: {step_loss:.3f} | '
        f'lr: {lr:.2e} | time (s): {step_time:.3f}',
        ranks=[0],
    )


def get_optimizer_grouped_parameters(
    model: BertForPreTraining,
) -> list[dict[str, Any]]:  # pragma: no cover
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    # configure the weight decay for bert models
    return [
        {
            'params': [
                p
                for n, p in param_optimizer
                if not any(v in n for v in no_decay)
            ],
            'weight_decay': 0.1,
        },
        {
            'params': [
                p for n, p in param_optimizer if any(v in n for v in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]


def get_optimizer(
    model: BertForPreTraining,
    name: Literal['lamb', 'adam'],
    lr: float,
) -> torch.optim.Optimizer:  # pragma: no cover
    optimizer_grouped_params = get_optimizer_grouped_parameters(model)

    if name == 'adam':
        optimizer = FusedAdam(
            optimizer_grouped_params,
            lr=lr,
            betas=[0.9, 0.95],
        )
    elif name == 'lamb':
        optimizer = FusedLAMB(optimizer_grouped_params, lr=lr)
    else:
        raise ValueError(f'Unknown optimizer: {name}')

    return optimizer


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = colossalai.get_default_parser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)

    if args.debug:
        colossalai.launch(
            config=args.config,
            rank=0,
            world_size=1,
            host='localhost',
            port=29400,
            backend='gloo',
        )
        args.local_rank = 0
    else:
        colossalai.launch_from_torch(config=args.config)

    gradient_accumulation = gpc.config.GLOBAL_BATCH_SIZE / (
        dist.get_world_size() * gpc.config.BATCH_SIZE
    )
    if gradient_accumulation != int(gradient_accumulation):
        raise ValueError(
            'Global batch size must be divisible by the product of the '
            'world size and batch size.',
        )
    gpc.config.update({'gradient_accumulation': int(gradient_accumulation)})

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()
    # TODO: https://github.com/hpcaitech/ColossalAI/issues/2109
    # logger.log_to_file(os.path.join(gpc.config.OUTPUT_DIR, 'logs'))

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
        ranks=[0],
    )

    model = bert.from_config(gpc.config.BERT_CONFIG)
    if gpc.config.get('GRADIENT_CHECKPOINTING', False):
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    optimizer = get_optimizer(model, gpc.config.OPTIMIZER, gpc.config.LR)
    criterion = BertPretrainingCriterion(gpc.config.BERT_CONFIG['vocab_size'])
    scheduler = PolynomialWarmupLR(
        optimizer,
        gpc.config.STEPS,
        gpc.config.WARMUP_STEPS,
        power=0.5,
    )

    engine, _, _, scheduler = colossalai.initialize(
        model,
        optimizer,
        criterion=criterion,
        lr_scheduler=scheduler,
        # Pass fake iterable for train_dataloader because setting
        # gradient_accumulation requires initialize to have a dataloader
        train_dataloader=list(range(0, 1024)),
    )

    micro_step = 0
    global_step = 0
    epoch = 0

    global_step_timer = colossalai.utils.Timer()
    step_loss = 0.0

    while global_step < gpc.config.STEPS:
        dataset = sharded_dataset(
            gpc.config.DATA_DIR,
            gpc.config.BATCH_SIZE,
            seed=gpc.config.SEED,
        )
        for batch in dataset:
            if micro_step % gpc.config.gradient_accumulation == 0:
                global_step_timer.start()

            batch = Batch(*[t.cuda() for t in batch])
            engine.zero_grad()
            output = engine(
                input_ids=batch.input_ids,
                token_type_ids=batch.segment_ids,
                attention_mask=batch.input_mask,
            )
            loss = engine.criterion(
                output.prediction_logits,
                batch.masked_labels,
                output.seq_relationship_logits,
                batch.next_sentence_labels,
            )
            engine.backward(loss)
            engine.step()

            step_loss += loss.float().item()
            micro_step += 1

            if micro_step % gpc.config.gradient_accumulation == 0:
                global_step += 1
                global_step_timer.stop(keep_in_history=True)
                log_step(
                    logger,
                    epoch=epoch,
                    global_step=global_step,
                    step_loss=step_loss / gpc.config.gradient_accumulation,
                    step_time=global_step_timer.get_elapsed_time(),
                    lr=scheduler.get_last_lr()[0],
                )
                step_loss = 0.0

            # Wait to step scheduler until after we have logged current LR
            scheduler.step()

            if global_step >= gpc.config.STEPS:
                # Break out of dataset loop, then while loop will also break
                break

        epoch += 1

    logger.info(
        'Training completed. Avg step time (s): '
        f'{global_step_timer.get_history_mean()}',
        ranks=[0],
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
