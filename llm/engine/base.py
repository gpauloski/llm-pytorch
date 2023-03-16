from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    """Base optimizer wrapper.

    Compared to a standard PyTorch [`Optimizer`][torch.optim.Optimizer],
    this optimizer adds the
    [`backward()`][llm.engine.base.BaseOptimizer.backward] that is used instead
    of directly calling `loss.backward()`. This is needed so the various
    optimizer wrappers can adjust how the loss is computed.

    Otherwise, this optimizer behaves identically to the wrapped optimizer.

    Args:
        optimizer: Standard PyTorch optimizer to wrap.
    """

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
        """Perform an optimization step."""
        self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        """Zero the gradients of optimized parameters."""
        self._optimizer.zero_grad(*args, **kwargs)

    def backward(self, loss: torch.Tensor) -> None:
        """Perform a backward pass using the loss."""
        loss.backward()
