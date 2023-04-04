from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from llm.config import Config
from llm.config import flatten_mapping
from llm.config import flattened_config
from llm.config import load_config


def test_config_attr_access() -> None:
    d = {'a': 1, 'b': 2}
    config = Config(**d)

    assert config == d
    assert config.a == d['a']
    assert config.b == d['b']

    with pytest.raises(AttributeError, match='c'):
        config.c  # noqa: B018

    config.a = 2
    assert config.a != d['a']

    config.c = 2
    assert config.c == 2


def test_config_nested_dict() -> None:
    d = {'a': {'b': 2}}
    config = Config(**d)

    assert config.a.b == 2


def test_config_from_existing() -> None:
    config = Config({'a': 1, 'b': 2})
    assert config.a == 1

    config = Config([('a', 1), ('b', 2)])
    assert config.a == 1


def test_flattened_config() -> None:
    assert flattened_config({'a': 1}) == {'a': 1}

    config = Config(**{'a': 1})
    assert flattened_config(config) == {'a': 1}

    # test get world size
    with mock.patch(
        'torch.distributed.is_initialized',
        return_value=True,
    ), mock.patch('torch.distributed.get_world_size', return_value=4):
        assert flattened_config() == {'world_size': 4}

    # test flattening and removing non-supported types
    config = Config(**{'a': {'b': 1, 'c': [1, 2]}})
    assert flattened_config(config) == {'a_b': 1}


@pytest.mark.parametrize(
    'in_,out,kwargs',
    (
        ({}, {}, {}),
        ({'a': 1}, {'a': 1}, {}),
        ({'a': {'b': 1}}, {'a_b': 1}, {}),
        ({'a': {'b': 1}}, {'a-b': 1}, {'sep': '-'}),
        ({'a': {'b': 1}}, {'x_a_b': 1}, {'parent': 'x'}),
        ({'a': {'b': {'c': 1}}}, {'a_b_c': 1}, {}),
        ({'a': {'b': 1}, 'c': 2}, {'a_b': 1, 'c': 2}, {}),
    ),
)
def test_flatten_mapping(
    in_: dict[str, Any],
    out: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    assert flatten_mapping(in_, **kwargs) == out


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
