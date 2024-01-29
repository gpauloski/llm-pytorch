"""Utilities for easy automatic mixed precision training."""

from __future__ import annotations

from typing import Any
from typing import Callable

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from llm.engine.base import BaseOptimizer


class AMPCriterion(torch.nn.Module):
    """Wrap a loss function for AMP training.

    Args:
        criterion: Loss function to wrap.
        autocast: Autocast context manager to compute loss inside.
    """

    def __init__(
        self,
        criterion: torch.nn.Module,
        autocast: torch.autocast,
    ) -> None:
        super().__init__()
        self._criterion = criterion
        self._autocast = autocast

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the loss inside the autocast."""
        with self._autocast:
            return self._criterion(*args, **kwargs)


class AMPModel(torch.nn.Module):
    """Wrap a model for AMP training.

    Args:
        model: Model to wrap.
        autocast: Autocast context manager to compute loss inside.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        autocast: torch.autocast,
    ) -> None:
        super().__init__()
        self._model = model
        self._autocast = autocast

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Perform a forward pass inside the autocast."""
        with self._autocast:
            return self._model(*args, **kwargs)


class AMPOptimizer(BaseOptimizer):
    """Wrap an optimizer for AMP training.

    Args:
        model: Model being optimized.
        optimizer: Optimizer to wrap.
        scaler: Gradient scaler.
        max_norm: Optionally clip gradient norm.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        max_norm: float | None = None,
    ) -> None:
        super().__init__(optimizer)
        self._model = model
        self._scaler = scaler
        self._max_norm = max_norm

    def state_dict(self) -> dict[str, Any]:
        """Dictionary containing references to the whole state of the module.

        Includes the state of the `grad_scaler`.
        """
        state_dict = self._optimizer.state_dict()
        assert 'grad_scaler' not in state_dict
        state_dict['grad_scaler'] = self._scaler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Copy the state into this module."""
        scaler_state_dict = state_dict.pop('grad_scaler', None)
        if scaler_state_dict is not None:  # pragma: no branch
            self._scaler.load_state_dict(scaler_state_dict)
        self._optimizer.load_state_dict(state_dict)

    def backward(self, loss: torch.Tensor) -> None:
        """Perform a backward pass and correctly scale the loss."""
        self._scaler.scale(loss).backward()

    def step(self, closure: Callable[[], float] | None = None) -> None:
        """Perform an optimization using the gradient scaler."""
        if self._max_norm is not None:
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._max_norm,
            )
        self._scaler.step(self._optimizer)
        self._scaler.update()


def initialize(
    model: torch.nn.Module,
    optimizer: Optimizer,
    criterion: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
    max_norm: float | None = None,
    **kwargs: Any,
) -> tuple[AMPModel, AMPOptimizer, AMPCriterion]:
    """Initialize AMP training.

    Args:
        model: Model being optimized.
        optimizer: Optimizer to wrap.
        criterion: Loss function to wrap.
        dtype: Data type to perform mixed precision in. Typically
            `torch.float16` or `torch.bfloat16`.
        max_norm: Optionally clip gradient norm.
        kwargs: Additional keyword arguments to pass to the
            [`GradScaler`][torch.cuda.amp.GradScaler].

    Returns:
        A tuple of the wrapped model, optimizer, and loss.
    """
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    autocast = torch.autocast(device, dtype=dtype)

    # GradScaler only works on CUDA tensors so we disable on CPU
    scaler = GradScaler(**kwargs, enabled=device == 'cuda')

    return (
        AMPModel(model, autocast),
        AMPOptimizer(model, optimizer, scaler, max_norm),
        AMPCriterion(criterion, autocast),
    )
