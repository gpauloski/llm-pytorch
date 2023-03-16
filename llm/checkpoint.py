from __future__ import annotations

import pathlib
import re
from typing import Any
from typing import NamedTuple

import torch

CHECKPOINT_NAME_RE = re.compile(r'^global_step_(\d+).pt$')


class Checkpoint(NamedTuple):
    """Data loaded from a checkpoint."""

    filepath: str
    """Filepath of checkpoint."""
    global_step: int
    """Global step of checkpoint."""
    model_state_dict: dict[Any, Any]
    """Model state dictionary."""
    optimizer_state_dict: dict[Any, Any] | None
    """Optimizer state dictionary."""
    scheduler_state_dict: dict[Any, Any] | None
    """Scheduler state dictionary."""
    kwargs: dict[str, Any]
    """Additional keyword arguments stored in the checkpoint."""


def load_checkpoint(
    checkpoint_dir: str | pathlib.Path,
    global_step: int | None = None,
    map_location: Any = None,
) -> Checkpoint | None:
    """Load checkpoint from directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        global_step: Global step checkpoint to load. If `None`,
            loads the latest checkpoint.
        map_location: Optional map_location to pass to
            [`torch.load()`][torch.load].

    Returns:
        Checkpoint or `None` if no checkpoint was found.

    Raises:
        OSError: If `checkpoint_dir` does not exist.
        OSError: If `global_step` is specified but the file does not exist.
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

    Saves the checkpoint as `{checkpoint_dir}/global_step_{global_step}.py`.

    Args:
        checkpoint_dir: Directory to save checkpoint to.
        global_step: Training step used as the key for checkpoints.
        model: Model to save state_dict of.
        optimizer: Optional optimizer to save state_dict of.
        scheduler: Optional scheduler to save state_dict of.
        kwargs: Additional key-value pairs to add to the checkpoint.
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
