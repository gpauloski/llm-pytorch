from __future__ import annotations

import logging
import pathlib
import sys
from collections.abc import Iterable
from typing import Any
from typing import Union

import torch.distributed as dist
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

HParamT = Union[bool, float, int, str, None]
"""Supported Hyperparameter types (i.e., JSON types)."""


def create_summary_writer(
    tensorboard_dir: str,
    hparam_dict: dict[str, HParamT] | None = None,
    metrics: list[str] | None = None,
    **writer_kwargs: Any,
) -> SummaryWriter:
    """Create a SummaryWriter instance for the run annotated with hyperparams.

    https://github.com/pytorch/pytorch/issues/37738#issuecomment-1124497827

    Args:
        tensorboard_dir: TensorBoard run directory.
        hparam_dict: Optional hyperparam dictionary to log alongside
            metrics.
        metrics: Optional list of metric tags that will be used with
            [`writer.add_scalar()`][torch.utils.tensorboard.writer.SummaryWriter.add_scalar]
            (e.g., `['train/loss', 'train/lr']`). Must be provided if
            `hparam_dict` is provided.
        writer_kwargs: Additional keyword arguments to pass to
            `SummaryWriter`.

    Returns:
        Summary writer instance.
    """
    writer = SummaryWriter(tensorboard_dir, **writer_kwargs)

    if hparam_dict is not None and metrics is not None:
        metric_dict = dict.fromkeys(metrics, 0)
        exp, ssi, sei = hparams(hparam_dict, metric_dict=metric_dict)
        assert writer.file_writer is not None
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)

    return writer


def get_filepaths(
    directory: pathlib.Path | str,
    extensions: list[str] | None = None,
    recursive: bool = False,
) -> list[str]:
    """Get list of filepaths in directory.

    Note:
        Only files (not sub-directories will be returned. Though
        sub-directories will be recursed into if `recursive=True`.

    Args:
        directory: Pathlike object with the directory to search.
        extensions: Pptionally only return files that match these extensions.
            Each extension should include the dot. E.g., `['.pdf', '.txt']`.
            Match is case sensitive.
        recursive: Recursively search sub-directories.

    Returns:
        List of string paths of files in the directory.
    """
    directory = pathlib.Path(directory)
    glob = '**/*' if recursive else '*'

    files = [path for path in directory.glob(glob) if path.is_file()]
    if extensions is not None:
        files = [path for path in files if path.suffix in extensions]
    return [str(path) for path in files]


def gradient_accumulation_steps(
    global_batch_size: int,
    local_batch_size: int,
    world_size: int,
) -> int:
    """Compute the gradient accumulation steps from the configuration.

    Args:
        global_batch_size: Target global/effective batch size.
        local_batch_size: Per rank batch size.
        world_size: Number of ranks.

    Returns:
        Gradient accumulation steps needed to achieve the `global_batch_size`.

    Raises:
        ValueError: If the resulting gradient accumulation steps would be
            fractional.
    """
    effective_batch = local_batch_size * world_size

    if global_batch_size % effective_batch != 0:
        raise ValueError(
            f'The global batch size ({global_batch_size}) must be evenly '
            'divisible by the product of the local batch size '
            f'({local_batch_size}) and the world size ({world_size}).',
        )

    return global_batch_size // effective_batch


class DistributedFilter(logging.Filter):
    """Custom filter that allows specifying ranks to print to.

    Example:
        ```python
        logger.info('My log message', extra={'ranks': [0, 2]})
        ```
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ranks = getattr(record, 'ranks', None)
        if dist.is_initialized():
            record.msg = f'[rank {dist.get_rank()}] {record.msg}'
            return ranks is None or dist.get_rank() in ranks
        else:
            return True


def init_logging(
    level: int | str = logging.INFO,
    logfile: pathlib.Path | str | None = None,
    rich: bool = False,
    distributed: bool = False,
) -> None:
    """Configure global logging.

    Args:
        level: Default logging level.
        logfile: Optional path to write logs to.
        rich: Use rich for pretty stdout logging.
        distributed: Configure distributed formatters and filters.
    """
    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(levelname)s (%(name)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    stdout_handler: logging.Handler
    if rich:
        stdout_handler = RichHandler(rich_tracebacks=True)
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [stdout_handler]

    if logfile is not None:
        path = pathlib.Path(logfile).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    if distributed:
        filter_ = DistributedFilter()
        for handler in handlers:
            handler.addFilter(filter_)

    logging.basicConfig(
        level=level,
        format='%(message)s',
        datefmt='[%X]',
        handlers=handlers,
    )


def log_step(
    logger: logging.Logger,
    step: int,
    *,
    fmt_str: str | None = None,
    log_level: int = logging.INFO,
    ranks: Iterable[int] = (0,),
    skip_tensorboard: Iterable[str] = (),
    tensorboard_prefix: str = 'train',
    writer: SummaryWriter | None = None,
    **kwargs: Any,
) -> None:
    """Log a training step.

    Args:
        logger: Logger instance to log to.
        step: Training step.
        fmt_str: Format string used to format parameters for logging.
        log_level: Level to log the parameters at.
        ranks: Ranks to log on (default to rank 0 only).
        skip_tensorboard: List of parameter names to skip logging to
            TensorBoard.
        tensorboard_prefix: Prefix for TensorBoard parameters.
        writer: TensorBoard summary writer.
        kwargs: Additional keyword arguments to log.
    """
    values = {'step': step, **kwargs}
    if fmt_str is not None:
        msg = fmt_str.format(**values)
    else:
        msg = ' | '.join(f'{name}: {value}' for name, value in values.items())

    logger.log(log_level, msg, extra={'ranks': ranks})
    if writer is not None:
        for name, value in kwargs.items():
            if name not in skip_tensorboard:
                writer.add_scalar(f'{tensorboard_prefix}/{name}', value, step)
