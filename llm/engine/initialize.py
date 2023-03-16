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
    dtype: torch.dtype | None = None,
    max_norm: float | None = None,
    **kwargs: Any,
) -> tuple[
    torch.nn.Module,
    BaseOptimizer,
    torch.nn.Module,
    torch.optim.lr_scheduler._LRScheduler,
]:
    """Enable advanced training features.

    This method allows you to easily wrap your training objects with
    transparent wrappers that enable advanced training features.

    Args:
        model: Model being trained.
        optimizer: Training optimizer.
        criterion: Training loss function.
        scheduler: LR scheduler.
        accumulation_steps: Number of forward/backward passes between
            optimizer steps.
        dtype: Optional data type for mixed precision training.
        max_norm: Optional maximum norm of gradients to clip to.
        kwargs: Keyword arguments to pass to the gradient scaler.

    Returns:
        Tuple of the wrapped model, optimizer, loss, and scheduler.
    """
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
            device_ids=[local_rank] if local_rank is not None else None,
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
