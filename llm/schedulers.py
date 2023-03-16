"""Custom learning rate schedules."""
from __future__ import annotations

import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupLR(_LRScheduler):
    """Linear warmup and decay LR scheduler.

    Source: [ColossalAI](https://github.com/hpcaitech/ColossalAI/blob/039b0c487bb7d380201d5d6aa6aa63e5d58c329b/colossalai/nn/lr_scheduler/linear.py){target=_blank}

    Args:
        optimizer: Optimizer to adjust learning rate of.
        total_steps: Total training steps.
        warmup_steps: Steps to linearly warmup the learning rate.
        last_epoch: Optional last epoch.
    """  # noqa: E501

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute the current learning rate."""
        if self.last_epoch < self.warmup_steps:
            factor = (self.last_epoch + 1) / (self.warmup_steps + 1)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            factor = self.total_steps - self.last_epoch
            factor = factor / (self.total_steps - self.warmup_steps)
            return [base_lr * factor for base_lr in self.base_lrs]
