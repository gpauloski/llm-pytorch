from __future__ import annotations

import logging
import os
from typing import Any

import torch

from llm.engine import accumulation
from llm.engine import amp
from llm.engine.base import BaseOptimizer

logger = logging.getLogger(__name__)


def initialize(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accumulation_steps: int = 1,
    dtype: torch.float16 | torch.bfloat16 | None = None,
    max_norm: float | None = None,
    **kwargs: Any,
) -> tuple[
    torch.nn.Module,
    BaseOptimizer,
    torch.nn.Module,
    torch.optim.lr_scheduler._LRScheduler,
]:
    if torch.cuda.is_available():
        logger.debug(
            f'Moving model to model to cuda:{torch.cuda.current_device()}.',
            extra={'ranks': [0]},
        )
        model.cuda()

    if torch.distributed.is_initialized():
        local_rank = (
            int(os.environ['LOCAL_RANK'])
            if torch.cuda.is_available()
            else None
        )
        logger.debug(
            'Wrapping model with DistributedDataParallel with '
            f'local_rank {local_rank}',
            extra={'ranks': [0]},
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    optimizer = BaseOptimizer(optimizer)

    if dtype is not None:
        logger.debug(
            f'Initializing model for AMP training with dtype {dtype}',
            extra={'ranks': [0]},
        )
        model, optimizer, criterion = amp.initialize(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dtype=dtype,
            max_norm=max_norm,
            **kwargs,
        )

    if accumulation_steps > 1:
        logger.debug(
            'Initializing model gradient accumulation steps = '
            f'{accumulation_steps}',
            extra={'ranks': [0]},
        )
        optimizer, scheduler = accumulation.initialize(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
        )

    return (model, optimizer, criterion, scheduler)
