from __future__ import annotations

from typing import Any
from typing import Callable

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer


class AMPCriterion(torch.nn.Module):
    def __init__(
        self,
        criterion: torch.nn.Module,
        autocast: torch.autocast,
    ) -> None:
        super().__init__()
        self._criterion = criterion
        self._autocast = autocast

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self._autocast:
            return self._criterion(*args, **kwargs)


class AMPModel(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        autocast: torch.autocast,
    ) -> None:
        super().__init__()
        self._model = model
        self._autocast = autocast

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        with self._autocast:
            return self._model(*args, **kwargs)


class AMPOptimizer(Optimizer):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        max_norm: float | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._scaler = scaler
        self._max_norm = max_norm

    def state_dict(self) -> dict[str, Any]:
        return {
            '_optimizer': self._optimizer.state_dict(),
            '_scaler': self._scaler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict['_optimizer'])
        self._scaler.load_state_dict(state_dict['_scaler'])

    def backward(self, loss: torch.Tensor) -> None:
        self._scaler.scale(loss).backward()

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> None:
        if self._max_norm is not None:
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                self._max_norm,
            )
        self._scaler.step(self._optimizer, closure=closure)
        self._scaler.update()

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad(set_to_none)


def initialize(
    model: torch.nn.Module,
    optimizer: Optimizer,
    criterion: torch.nn.loss._Loss,
    dtype: torch.float16 | torch.bfloat16 = torch.float16,
    max_norm: float | None = None,
    **kwargs: Any,
) -> tuple[AMPModel, AMPOptimizer, AMPCriterion]:
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    autocast = torch.autocast(device, dtype=dtype)

    # GradScaler only works on CUDA tensors so we disable on CPU
    scaler = GradScaler(**kwargs, enabled=device == 'cuda')

    return (
        AMPModel(model, autocast),
        AMPOptimizer(model, optimizer, scaler, max_norm),
        AMPCriterion(criterion, autocast),
    )
