from __future__ import annotations

import pytest
import torch

from llm.optimizers import get_optimizer


@pytest.fixture()
def model() -> torch.nn.Module:
    return torch.nn.Linear(1, 1)


def test_adam_optimizer(model: torch.nn.Module) -> None:
    # Adam should always work because we have a fall back
    assert isinstance(
        get_optimizer('adam', model.parameters(), 0.1),
        torch.optim.Optimizer,
    )


def test_unknown_optimizer(model: torch.nn.Module) -> None:
    with pytest.raises(ValueError, match='xyz'):
        get_optimizer('xyz', model.parameters(), 0.1)  # type: ignore[arg-type]
