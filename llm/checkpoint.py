from __future__ import annotations

import pathlib
import re
from typing import Any
from typing import NamedTuple

import torch

CHECKPOINT_NAME_RE = re.compile(r'^global_step_(\d+).pt$')


class Checkpoint(NamedTuple):
    filepath: str
    global_step: int
    model_state_dict: dict[Any, Any]
    optimizer_state_dict: dict[Any, Any] | None
    scheduler_state_dict: dict[Any, Any] | None
    kwargs: dict[str, Any]


def load_checkpoint(
    checkpoint_dir: str | pathlib.Path,
    global_step: int | None = None,
    map_location: Any = None,
) -> Checkpoint | None:
    """Load checkpoint from directory.

    Args:
        checkpoint_dir (str): directory containing checkpoint files.
        global_step (int, optional): global step checkpoint to load. If None,
            loads the latest checkpoint.
        map_location: optional map_location to pass to torch.load().

    Returns:
        Checkpoint or None if no checkpoint was found.

    Raises:
        OSError:
            if checkpoint_dir does not exist.
        OSError:
            if global_step is specified but the file does not exist.
    """
    dir_path = pathlib.Path(checkpoint_dir)
    if not dir_path.is_dir():
        raise OSError(f'Checkpoint directory {checkpoint_dir} does not exist.')

    if global_step is None:
        checkpoints = {
            match.group(1): str(p)
            for p in dir_path.iterdir()
            if (match := CHECKPOINT_NAME_RE.fullmatch(p.name)) is not None
        }
        if len(checkpoints) == 0:
            return None
        steps = [int(key) for key in checkpoints]
        global_step = max(steps)

    assert global_step is not None

    checkpoint_path = dir_path / f'global_step_{global_step}.pt'
    if not checkpoint_path.is_file():
        raise OSError(f'Checkpoint named {checkpoint_path} does not exist.')

    state_dict = torch.load(checkpoint_path, map_location=map_location)

    model_state_dict = state_dict.pop('model')
    optimizer_state_dict = state_dict.pop('optimizer', None)
    scheduler_state_dict = state_dict.pop('scheduler', None)

    return Checkpoint(
        filepath=str(checkpoint_path),
        global_step=global_step,
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        kwargs=state_dict,
    )


def save_checkpoint(
    checkpoint_dir: str | pathlib.Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    **kwargs: Any,
) -> None:
    """Save checkpoint to directory.

    Saves the checkpoint as {checkpoint_dir}/global_step_{global_step}.py.

    Args:
        checkpoint_dir (str): directory to save checkpoint to.
        global_step (int): training step used as the key for checkpoints.
        model: model to save state_dict of.
        optimizer: optional optimizer to save state_dict of.
        scheduler: optional scheduler to save state_dict of.
        kwargs: additional key-value pairs to add to the checkpoint.
    """
    state_dict = {'model': model.state_dict(), **kwargs}
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()

    dir_path = pathlib.Path(checkpoint_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dir_path / f'global_step_{global_step}.pt'

    torch.save(state_dict, checkpoint_path)
