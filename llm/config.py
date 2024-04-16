from __future__ import annotations

import collections
import inspect
import pathlib
import types
from collections.abc import Iterable
from collections.abc import Mapping
from importlib.machinery import SourceFileLoader
from typing import Any
from typing import Union

import torch.distributed as dist

HParamT = Union[bool, float, int, str, None]
"""Supported Hyperparameter types (i.e., JSON types)."""


class Config(dict[str, Any]):
    """Dict-like configuration class with attribute access.

    Example:
        ```python
        >>> from llm.config import Config
        >>> config = Config({'a': 1, 'b': 2})
        >>> config.a
        1
        >>> config = Config(a=1, b={'c': 2})
        >>> config.b.c
        2
        ```

    Args:
        mapping: Initial mapping or iterable of tuples of key-value pairs.
        kwargs: Keywords arguments to add a key-value pairs to the config.
    """

    def __init__(
        self,
        mapping: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
        /,
        **kwargs: Any,
    ):
        if mapping is not None:
            if isinstance(mapping, Mapping):
                mapping = mapping.items()
            for key, value in mapping:
                self.__setattr__(key, value)

        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getitem__(key)
        except KeyError:
            raise AttributeError(
                f'{key}... are you missing `{key}` in your config?',
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            value = Config(**value)
        super().__setitem__(key, value)


def flattened_config(
    config: dict[str, Any] | Config | None = None,
) -> dict[str, HParamT]:
    """Convert a config to a flat JSONable dictionary.

    Note:
        If
        [`torch.distributed.is_initialized()`][torch.distributed.is_initialized],
        the `world_size` will be added to the config.

    Args:
        config: Optional starting config.

    Returns:
        Flat dictionary containing only `bool`, `float`, `int`, `str`, or
        `None` values.
    """
    if config is None:
        config = {}

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
        ```python
        >>> flatten_mapping({'a': {'b_c': 1}, 'a_b': {'c': 2}})
        {'a_b_c': 2}
        ```

    Args:
        d: Input mapping. All keys and nested keys must by strings.
        parent: Parent key to prepend to top-level keys in `d`.
        sep: Separator between keys.

    Returns:
        Flattened dictionary.
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


def load_config(filepath: pathlib.Path | str) -> Config:
    """Load Python file as a [`Config`][llm.config.Config].

    Note:
        Attributes starting with `_`, modules, classes, functions, and
        builtins will not be loaded from the Python file.

    Args:
        filepath: Python file to load.

    Returns:
        Configuration attributes loaded from the Python file.
    """
    filepath = pathlib.Path(filepath).absolute()
    if not filepath.exists():
        raise OSError(f'{filepath} does not exist.')
    elif filepath.suffix != '.py':
        raise ValueError(
            f'{filepath} is not a Python file. Only .py files are supported.',
        )

    loader = SourceFileLoader(fullname=filepath.stem, path=str(filepath))
    module = types.ModuleType(loader.name)
    loader.exec_module(module)

    attrs: dict[str, Any] = {}
    for key, value in module.__dict__.items():
        if not (
            key.startswith('_')
            or inspect.ismodule(value)
            or inspect.isclass(value)
            or inspect.isfunction(value)
            or inspect.isbuiltin(value)
        ):
            attrs[key] = value

    return Config(**attrs)
