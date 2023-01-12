from __future__ import annotations

import pathlib
from typing import Any
from unittest import mock

import pytest

from llm.utils import create_summary_writer
from llm.utils import flatten_mapping
from llm.utils import flattened_config
from llm.utils import gradient_accumulation_steps


def test_create_summary_writer(tmp_path: pathlib.Path) -> None:
    writer = create_summary_writer(str(tmp_path))
    writer.add_scalar('test', 0.1)
    writer.close()

    with pytest.raises(TypeError):
        # missing_kwargs is passed to SummaryWriter which should error
        create_summary_writer(str(tmp_path), missing_kwargs=True)


def test_create_summary_writer_hparams(tmp_path: pathlib.Path) -> None:
    writer = create_summary_writer(
        str(tmp_path),
        {'lr': 0.01, 'epochs': 0.1},
        ['train/loss', 'train/lr'],
    )
    writer.add_scalar('train/loss', 0.1)
    writer.close()


def test_grad_accumulation_steps() -> None:
    assert gradient_accumulation_steps(8, 4, 2) == 1
    assert gradient_accumulation_steps(16, 4, 2) == 2
    assert gradient_accumulation_steps(32, 1, 1) == 32

    with pytest.raises(ValueError):
        gradient_accumulation_steps(8, 3, 2)


def test_flattened_config() -> None:
    with mock.patch('llm.utils.gpc') as mock_gpc:
        mock_gpc.config = {}
        assert flattened_config({'a': 1}) == {'a': 1}

        mock_gpc.config = None
        assert flattened_config({'a': 1}) == {'a': 1}

        # gpc overrides caller base config
        mock_gpc.config = {'a': 2}
        assert flattened_config({'a': 1}) == {'a': 2}

        # test get world size
        mock_gpc.config = {}
        with mock.patch(
            'torch.distributed.is_initialized',
            return_value=True,
        ), mock.patch('torch.distributed.get_world_size', return_value=4):
            assert flattened_config() == {'world_size': 4}

        # test flattening and removing non-supported types
        mock_gpc.config = {'a': {'b': 1, 'c': [1, 2]}}
        assert flattened_config() == {'a_b': 1}


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
