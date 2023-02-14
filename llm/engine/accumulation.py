from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler

from llm.engine.base import BaseOptimizer


class GradientAccumulationOptimizer(BaseOptimizer):
    def __init__(
        self,
        optimizer: BaseOptimizer,
        model: torch.nn.Module,
        accumulation_steps: int,
    ) -> None:
        super().__init__(optimizer)
        self._accumulation_steps = accumulation_steps
        self._accumulation_step = 0
        self._model = model

    def accumulation_boundary(self) -> bool:
        return self._accumulation_step == 0

    def backward(self, loss: torch.Tensor) -> None:
        self._accumulation_step += 1

        context = (
            self._model.no_sync()  # type: ignore[operator]
            if (
                self._accumulation_step < self._accumulation_steps
                and isinstance(self._model, DistributedDataParallel)
            )
            else contextlib.nullcontext()
        )

        with context:
            scaled_loss = loss / self._accumulation_steps
            scaled_loss.backward()

    def step(self, *args: Any, **kwargs: Any) -> None:
        if self._accumulation_step == self._accumulation_steps:
            self._accumulation_step = 0
            self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        if self._accumulation_step == 0:
            self._optimizer.zero_grad(*args, **kwargs)


class GradientAccumulationLRScheduler(_LRScheduler):
    def __init__(
        self,
        scheduler: _LRScheduler,
        accumulation_steps: int,
    ) -> None:
        self._accumulation_steps = accumulation_steps
        self._accumulation_step = 0
        self._scheduler = scheduler

    def __getattr__(self, attr: str) -> Any:
        assert attr != 'step'
        return getattr(self._scheduler, attr)

    def step(self, epoch: int | None = None) -> None:
        self._accumulation_step += 1
        if self._accumulation_step == self._accumulation_steps:
            self._accumulation_step = 0
            self._scheduler.step(epoch)


def initialize(
    model: torch.nn.Module,
    optimizer: BaseOptimizer,
    scheduler: _LRScheduler,
    accumulation_steps: int = 1,
) -> tuple[GradientAccumulationOptimizer, GradientAccumulationLRScheduler]:
    return (
        GradientAccumulationOptimizer(optimizer, model, accumulation_steps),
        GradientAccumulationLRScheduler(scheduler, accumulation_steps),
    )
