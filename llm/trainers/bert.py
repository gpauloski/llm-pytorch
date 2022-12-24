from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any
from typing import Literal

import colossalai
import torch
import torch.distributed as dist
from colossalai.core import global_context as gpc
from colossalai.nn.optimizer import FusedAdam
from colossalai.nn.optimizer import FusedLamb
from transformers import BertForPreTraining

from llm.datasets.nvidia import sharded_dataset


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
        optimizer = FusedLamb(optimizer_grouped_params, lr=lr)
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

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()
    # TODO: https://github.com/hpcaitech/ColossalAI/issues/2109
    # logger.log_to_file(os.path.join(gpc.config.OUTPUT_DIR, 'logs'))

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
    )

    if gpc.config.WORKERS != dist.get_world_size():
        logger.warning(
            f'Expected number of workers in the config ({gpc.config.WORKERS}) '
            f'does not match the found number of workers '
            f'({dist.get_world_size()}). This can result in incorrect '
            'gradient accumulation',
        )

    step = 0
    epoch = 0

    while step < gpc.config.STEPS:
        dataset = sharded_dataset(gpc.config.DATA_DIR, gpc.config.BATCH_SIZE)
        for batch in dataset:
            batch
            # forward
            # backward
            # loss
            # optimize

            step += 1
            if step >= gpc.config.STEPS:
                # Break out of dataset loop, then while loop will also break
                break

        epoch += 1

    logger.info('Training completed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
