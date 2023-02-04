from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    def __init__(self, optimizer: Optimizer):
        self._optimizer = optimizer

    @property
    def defaults(self) -> Any:
        return self._optimizer.defaults

    @property
    def param_groups(self) -> Any:
        return self._optimizer.param_groups

    def add_param_group(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:  # pragma: no cover
        self._optimizer.add_param_group(*args, **kwargs)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)

    def state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def step(self, *args: Any, **kwargs: Any) -> None:
        return self._optimizer.step(*args, **kwargs)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad(set_to_none)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
