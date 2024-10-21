from __future__ import annotations

import os
import pathlib
from collections.abc import Generator
from unittest import mock

import pytest

from llm.initialize import get_default_parser
from llm.initialize import initialize
from llm.initialize import initialize_from_args


@pytest.fixture
def distributed() -> Generator[None, None, None]:  # noqa: PT004
    with (
        mock.patch('torch.distributed.init_process_group'),
        mock.patch('torch.distributed.get_backend', return_value=True),
        mock.patch('torch.distributed.get_world_size', return_value=1),
    ):
        yield


def test_initialize_from_args(distributed, tmp_path: pathlib.Path) -> None:
    config_str = 'a = 1'
    config_path = tmp_path / 'config.py'
    with open(config_path, 'w') as f:
        f.write(config_str)

    parser = get_default_parser()
    args = parser.parse_args(['--config', str(config_path), '--debug'])

    config = initialize_from_args(args)
    assert config.a == 1
    assert os.environ['LOCAL_RANK'] == '0'
    del os.environ['LOCAL_RANK']


@pytest.mark.parametrize('cuda', (True, False))
@mock.patch.dict(os.environ, {'LOCAL_RANK': '0'})
def test_initialize_cuda(cuda, distributed) -> None:
    with (
        mock.patch('torch.cuda.is_available', return_value=cuda),
        mock.patch('torch.cuda.set_device') as mock_device,
    ):
        initialize(seed=42)
        assert mock_device.called == cuda
