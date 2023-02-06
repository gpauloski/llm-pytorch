from __future__ import annotations

import logging
import pathlib
from unittest import mock

import pytest

from llm.utils import DistributedFilter
from llm.utils import create_summary_writer
from llm.utils import gradient_accumulation_steps
from llm.utils import init_logging
from llm.utils import log_step


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


@pytest.mark.parametrize('rich', (True, False))
def test_logging(rich: bool) -> None:
    init_logging(rich=rich)

    logger = logging.getLogger('tests/utils::test_logging')
    logger.info('test')


def test_logging_file_handler(tmp_path: pathlib.Path) -> None:
    init_logging(logfile=tmp_path / 'log.txt')

    logger = logging.getLogger('tests/utils::test_logging_file_handler')
    logger.info('test')


def test_distributed_logging(caplog) -> None:
    caplog.set_level(logging.INFO)
    init_logging(distributed=True)

    # Pytest replaces the logging handlers so there is not much to test here
    logger = logging.getLogger('tests/utils::test_distributed_logging')

    filter_ = DistributedFilter()

    # Distributed not initialized
    logger.info('test', extra={'ranks': [1]})
    record = caplog.records.pop()
    assert filter_.filter(record)

    with mock.patch('torch.distributed.is_initialized', return_value=True):
        with mock.patch('torch.distributed.get_rank', return_value=0):
            logger.info('test')
            record = caplog.records.pop()
            assert filter_.filter(record)

            logger.info('test', extra={'ranks': [0]})
            record = caplog.records.pop()
            assert filter_.filter(record)

            logger.info('test', extra={'ranks': [1]})
            record = caplog.records.pop()
            assert not filter_.filter(record)


def test_log_step(caplog) -> None:
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger('tests/utils::test_log_step')

    for i in range(10):
        log_step(logger, i, log_level=logging.DEBUG)

    for record in caplog.records:
        assert record.levelno == logging.DEBUG
    assert len(caplog.records) == 10


def test_log_step_fmt(caplog) -> None:
    caplog.set_level(logging.INFO)
    logger = logging.getLogger('tests/utils::test_log_step_fmt')

    fmt_str = 'step={step}, lr={lr}, epoch={epoch}'
    log_step(logger, 0, lr=0.1, epoch=1, fmt_str=fmt_str)

    record = caplog.records[0]
    assert record.message == 'step=0, lr=0.1, epoch=1'


def test_log_step_summary_writer(tmp_path: pathlib.Path) -> None:
    logger = logging.getLogger('tests/utils::test_log_step_summary_writer')

    writer = create_summary_writer(str(tmp_path))
    log_step(
        logger,
        0,
        lr=0.1,
        epoch=1,
        time=0.2,
        writer=writer,
        skip_tensorboard=['time'],
        tensorboard_prefix='test',
    )
    writer.close()
