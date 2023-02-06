from __future__ import annotations

import collections
import inspect
import pathlib
from importlib.machinery import SourceFileLoader
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Union

import torch.distributed as dist

HParamT = Union[bool, float, int, str, None]


class Config(dict[str, Any]):
    """Dict-like configuration class with attribute access."""

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
    """Return flattened global config as JSON.

    Args:
        config (dict): optional starting config.

    Returns:
        flat dictionary containing only bool, float, int, str, or None values.
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


def load_config(filepath: pathlib.Path | str) -> Config:
    """Load python file as a Config.

    Args:
        filepath: python filepath to load.

    Returns:
        Config
    """
    filepath = pathlib.Path(filepath).absolute()
    if not filepath.exists():
        raise OSError(f'{filepath} does not exist.')
    elif filepath.suffix != '.py':
        raise ValueError(
            f'{filepath} is not a Python file. Only .py files are supported.',
        )

    module = SourceFileLoader(
        fullname=filepath.stem,
        path=str(filepath),
    ).load_module()

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
