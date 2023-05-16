"""Utilities for easy gradient accumulation training."""
from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler

from llm.engine.base import BaseOptimizer


class GradientAccumulationOptimizer(BaseOptimizer):
    """Optimizer wrapper for enabling gradient accumulation.

    This wrapper will skip calls to
    [`BaseOptimizer.step()`][llm.engine.base.BaseOptimizer.step] until
    `accumulation_steps` forward/backward passes have been performed.

    Args:
        optimizer: Optimizer to wrap.
        model: Model being optimized.
        accumulation_steps: Number of iterations between optimization steps.
    """

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
        """Return if the current step is an accumulation boundary.

        I.e., the last call to
        [`step()`][llm.engine.accumulation.GradientAccumulationOptimizer.step]
        resulted in an optimization step and no accumulation for the next step
        has started.
        """
        return self._accumulation_step == 0

    def backward(self, loss: torch.Tensor) -> None:
        """Perform a backward pass.

        Note:
            If `model` is a
            [`DistributedDataParallel`][torch.nn.parallel.DistributedDataParallel]
            instance, backward passes will be performed with
            [`no_sync()`][torch.nn.parallel.DistributedDataParallel.no_sync]
            during gradient accumulation steps.

        Args:
            loss: Loss to compute gradients with respect to.
        """
        self._accumulation_step += 1

        context = (
            self._model.no_sync()
            if (
                self._accumulation_step < self._accumulation_steps
                and isinstance(self._model, DistributedDataParallel)
            )
            else contextlib.nullcontext()
        )

        with context:
            scaled_loss = loss / self._accumulation_steps
            self._optimizer.backward(scaled_loss)

    def step(self, *args: Any, **kwargs: Any) -> None:
        """Perform an optimization step.

        This method is a no-op unless `accumulation_steps` have occurred.
        """
        if self._accumulation_step == self._accumulation_steps:
            self._accumulation_step = 0
            self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Zero the gradients of the wrapped optimizer."""
        if self._accumulation_step == 0:
            self._optimizer.zero_grad(*args, **kwargs)


class GradientAccumulationLRScheduler(_LRScheduler):
    """LR scheduler wrapper that accounts for gradient accumulation.

    This wrapper allows you to call `scheduler.step()` after every
    forward/backward pass and will correctly skip the call if
    it happens during a gradient accumulation period.

    Args:
        scheduler: LR scheduler to wrap.
        accumulation_steps: Number of iterations between optimization steps.
    """

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
        """Update the learning rate.

        This method is a no-op unless `accumulation_steps` have occurred.
        """
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
    """Initialize gradient accumulation training.

    Args:
        model: Model being optimized.
        optimizer: Optimizer to wrap.
        scheduler: LR scheduler to wrap.
        accumulation_steps: Number of iterations between optimization steps.

    Returns:
        The wrapped optimizer and LR scheduler.
    """
    return (
        GradientAccumulationOptimizer(optimizer, model, accumulation_steps),
        GradientAccumulationLRScheduler(scheduler, accumulation_steps),
    )
