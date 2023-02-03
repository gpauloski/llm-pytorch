from __future__ import annotations

import pathlib

import pytest

from llm.config import Config
from llm.config import load_config


def test_config_attr_access() -> None:
    d = {'a': 1, 'b': 2}
    config = Config(**d)

    assert config == d
    assert config.a == d['a']
    assert config.b == d['b']

    with pytest.raises(AttributeError, match='c'):
        config.c

    config.a = 2
    assert config.a != d['a']

    config.c = 2
    assert config.c == 2


def test_config_nested_dict() -> None:
    d = {'a': {'b': 2}}
    config = Config(**d)

    assert config.a.b == 2


def test_load_config(tmp_path: pathlib.Path) -> None:
    config_str = """\
from math import ceil

def double(x: int) -> int:
    return 2 * x

a = 1
b = 2
_c = 3

class Person:
    pass
"""
    filepath = tmp_path / 'config.py'
    with open(filepath, 'w') as f:
        f.write(config_str)

    config = load_config(filepath)
    assert config == {'a': 1, 'b': 2}


def test_load_config_missing_file(tmp_path: pathlib.Path) -> None:
    with pytest.raises(OSError, match='exist'):
        load_config(tmp_path / 'config.py')


def test_load_config_wrong_type(tmp_path: pathlib.Path) -> None:
    filepath = tmp_path / 'config.json'
    filepath.touch()
    with pytest.raises(ValueError, match='.py'):
        load_config(filepath)
