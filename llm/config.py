from __future__ import annotations

import inspect
import pathlib
from importlib.machinery import SourceFileLoader
from typing import Any


class Config(dict[str, Any]):
    """Dict-like configuration class with attribute access."""

    def __init__(self, **kwargs: Any):
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
