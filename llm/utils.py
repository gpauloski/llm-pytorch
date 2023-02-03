from __future__ import annotations

import collections
import logging
import pathlib
import sys
from collections.abc import Mapping
from typing import Any
from typing import Union

import torch.distributed as dist
from colossalai.core import global_context as gpc
from rich.logging import RichHandler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

HParamT = Union[bool, float, int, str, None]


def create_summary_writer(
    tensorboard_dir: str,
    hparam_dict: dict[str, HParamT] | None = None,
    metrics: list[str] | None = None,
    **writer_kwargs: Any,
) -> SummaryWriter:
    """Create a SummaryWriter instance for the run annotated with hyperparams.

    https://github.com/pytorch/pytorch/issues/37738#issuecomment-1124497827

    Args:
        tensorboard_dir (str): TensorBoard run directory.
        hparam_dict (dict): optional hyperparam dictionary to log alongside
            metrics (default: None).
        metrics (list[str]): optional list of metric tags that will be used
            with ``writer.add_scalar()`` (e.g., ['train/loss', 'train/lr']).
            Must be provided if ``hparam_dict`` is provided (default: None).
        **writer_kwargs: additional keyword arguments to pass to
            ``SummaryWriter``.

    Returns:
        SummaryWriter
    """
    writer = SummaryWriter(tensorboard_dir, **writer_kwargs)

    if hparam_dict is not None and metrics is not None:
        metric_dict = {metric: 0 for metric in metrics}
        exp, ssi, sei = hparams(hparam_dict, metric_dict=metric_dict)
        writer.file_writer.add_summary(exp)
        writer.file_writer.add_summary(ssi)
        writer.file_writer.add_summary(sei)

    return writer


def gradient_accumulation_steps(
    global_batch_size: int,
    local_batch_size: int,
    world_size: int,
) -> int:
    effective_batch = local_batch_size * world_size

    if global_batch_size % effective_batch != 0:
        raise ValueError(
            f'The global batch size ({global_batch_size}) must be evenly '
            'divisible by the product of the local batch size '
            f'({local_batch_size}) and the world size ({world_size}).',
        )

    return global_batch_size // effective_batch


def flattened_config(
    config: dict[str, HParamT] | None = None,
) -> dict[str, HParamT]:
    """Return flattened global config as JSON.

    Args:
        config (dict): optional starting config. Note that ``gpc.config``
            will override these values.

    Returns:
        flat dictionary containing only bool, float, int, str, or None values.
    """
    if config is None:
        config = {}

    if gpc.config is not None:
        config.update(gpc.config)

    if dist.is_initialized():
        config['world_size'] = dist.get_world_size()

    config = flatten_mapping(config)
    for key in list(config.keys()):
        if not isinstance(config[key], (bool, float, int, str, type(None))):
            del config[key]

    return config


def flatten_mapping(
    d: Mapping[str, Any],
    parent: str | None = None,
    sep: str = '_',
) -> dict[str, Any]:
    """Flatten mapping into dict by joining nested keys via a separator.

    Warning:
        This function does not check for key collisions. E.g.,
        >>> flatten_mapping({'a': {'b_c': 1}, 'a_b': {'c': 2}})
        {'a_b_c': 2}

    Args:
        d (Mapping): input mapping. All keys and nested keys must by strings.
        parent (str, optional): parent key to prepend to top-level keys in
            ``d`` (Default: None).
        sep (str): separator between keys (Default: '_').

    Returns:
        flat dictionary.
    """
    # https://stackoverflow.com/questions/6027558
    items: list[tuple[str, Any]] = []

    for key, value in d.items():
        new_key = f'{parent}{sep}{key}' if parent is not None else key
        if isinstance(value, collections.abc.Mapping):
            items.extend(flatten_mapping(value, new_key, sep).items())
        else:
            items.append((new_key, value))

    return dict(items)


class DistributedFilter(logging.Filter):
    """Custom filter that allows specifying ranks to print to."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
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
        level: default logging level.
        logfile: optional path to write logs to.
        rich: use rich for pretty stdout logging.
        distributed: configure distributed formatters and filters.
    """
    if rich:
        stdout_handler = RichHandler(rich_tracebacks=True)
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
    handlers: list[logging.Handler] = [stdout_handler]

    if logfile is not None:
        path = pathlib.Path(logfile).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path))

    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)03d] %(levelname)s (%(name)s): %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    for handler in handlers:
        handler.setFormatter(formatter)

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
