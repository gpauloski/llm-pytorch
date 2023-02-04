from __future__ import annotations

from typing import Any

import torch

from llm.engine import accumulation
from llm.engine import amp
from llm.engine.base import BaseOptimizer


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
    if dtype is not None:
        model, optimizer, criterion = amp.initialize(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dtype=dtype,
            max_norm=max_norm,
            **kwargs,
        )

    if accumulation_steps > 1:
        optimizer, scheduler = accumulation.initialize(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accumulation_steps=accumulation_steps,
        )

    if not isinstance(optimizer, BaseOptimizer):
        optimizer = BaseOptimizer(optimizer)

    return (model, optimizer, criterion, scheduler)
