from __future__ import annotations

import pathlib
from typing import Any

import pytest
import torch

from llm.checkpoint import load_checkpoint
from llm.checkpoint import save_checkpoint

MODEL = torch.nn.Linear(10, 10)
OPTIMIZER = torch.optim.SGD(MODEL.parameters(), 0.1)
SCHEDULER = torch.optim.lr_scheduler.LinearLR(OPTIMIZER)


def states_equal(
    state_a: dict[str, Any] | None,
    state_b: dict[str, Any] | None,
) -> bool:
    # Comparing the str representations is an easier way to check equality
    # but is not necessarily the fastest
    return str(state_a) == str(state_b)


@pytest.mark.parametrize(
    ('step', 'model', 'optimizer', 'scheduler', 'kwargs'),
    (
        (0, MODEL, None, None, {}),
        (1, MODEL, OPTIMIZER, SCHEDULER, {'phase': 1}),
    ),
)
def test_save_load(
    tmp_path: pathlib.Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    kwargs: dict[str, Any],
) -> None:
    save_checkpoint(tmp_path, step, model, optimizer, scheduler, **kwargs)
    ckpt = load_checkpoint(tmp_path, step)

    assert ckpt is not None
    assert step == ckpt.global_step
    assert states_equal(model.state_dict(), ckpt.model_state_dict)
    if optimizer is not None:
        assert states_equal(optimizer.state_dict(), ckpt.optimizer_state_dict)
    else:
        assert ckpt.optimizer_state_dict is None
    if scheduler is not None:
        assert states_equal(scheduler.state_dict(), ckpt.scheduler_state_dict)
    else:
        assert ckpt.scheduler_state_dict is None
    assert kwargs == ckpt.kwargs


def test_load_latest(tmp_path: pathlib.Path) -> None:
    save_checkpoint(tmp_path, 0, MODEL)
    save_checkpoint(tmp_path, 1, MODEL)

    ckpt = load_checkpoint(tmp_path)
    assert ckpt is not None
    assert ckpt.global_step == 1


def test_empty_dir(tmp_path: pathlib.Path) -> None:
    assert load_checkpoint(tmp_path) is None


def test_checkpoint_missing_dir(tmp_path: pathlib.Path) -> None:
    with pytest.raises(OSError, match='directory'):
        load_checkpoint(tmp_path / 'missing-dir')


def test_checkpoint_missing_checkpoint(tmp_path: pathlib.Path) -> None:
    with pytest.raises(OSError, match='1000'):
        load_checkpoint(tmp_path, 1000)
