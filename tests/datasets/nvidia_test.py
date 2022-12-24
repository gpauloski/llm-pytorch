from __future__ import annotations

import pathlib
import random
import tempfile
from collections.abc import Generator
from typing import Any
from typing import TypeVar

import h5py
import pytest

from llm.datasets.nvidia import Batch
from llm.datasets.nvidia import get_shard_filepaths
from llm.datasets.nvidia import load_dataset_from_shard
from llm.datasets.nvidia import NvidiaBertDataset
from llm.datasets.nvidia import sharded_dataset
from llm.datasets.nvidia import WorkerInitObj


NUM_FILE = 3
NUM_SAMPLES = 32
SEQ_LEN = 128

T = TypeVar('T', int, bool)


def generate_int_sample(seq_len: int) -> list[int]:
    return [random.randint(0, 30000) for _ in range(seq_len)]


def generate_bool_sample(seq_len: int) -> list[int]:
    return [random.choice((True, False)) for _ in range(seq_len)]


def generate_data(samples: int, seq_len: int) -> tuple[list[Any], ...]:
    return (
        [generate_int_sample(seq_len) for _ in range(samples)],
        [generate_bool_sample(seq_len) for _ in range(samples)],
        [generate_bool_sample(seq_len) for _ in range(samples)],
        [generate_int_sample(seq_len) for _ in range(samples)],
        [generate_int_sample(seq_len) for _ in range(samples)],
        generate_bool_sample(samples),
    )


def write_hdf5(
    filepath: pathlib.Path | str,
    input_ids: list[list[int]],
    input_mask: list[list[bool]],
    segment_ids: list[list[bool]],
    masked_lm_positions: list[list[int]],
    masked_lm_ids: list[list[int]],
    next_sentence_labels: list[int],
) -> None:
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('input_ids', data=input_ids, dtype='i4')
        f.create_dataset('input_mask', data=input_mask, dtype='i1')
        f.create_dataset('segment_ids', data=segment_ids, dtype='i1')
        f.create_dataset(
            'masked_lm_positions',
            data=masked_lm_positions,
            dtype='i4',
        )
        f.create_dataset('masked_lm_ids', data=masked_lm_ids, dtype='i4')
        f.create_dataset(
            'next_sentence_labels',
            data=next_sentence_labels,
            dtype='i1',
        )


@pytest.fixture(scope='module')
def data_dir() -> Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        for index in range(NUM_FILE):
            filepath = tmp_path / f'{index}.hdf5'
            data = generate_data(NUM_SAMPLES, SEQ_LEN)
            write_hdf5(filepath, *data)

        yield tmp_path


def test_worker_init() -> None:
    init = WorkerInitObj(0)
    init(3)


def test_get_shard_filepaths(data_dir: pathlib.Path) -> None:
    assert len(get_shard_filepaths(str(data_dir))) == NUM_FILE

    tmp_file = data_dir / 'other'
    tmp_file.touch()
    assert len(get_shard_filepaths(str(data_dir))) == NUM_FILE
    tmp_file.unlink()

    tmp_file = data_dir / 'other.h5'
    tmp_file.touch()
    assert len(get_shard_filepaths(str(data_dir))) == NUM_FILE + 1
    tmp_file.unlink()


def test_dataset(data_dir: pathlib.Path) -> None:
    shard = str(next(data_dir.iterdir()))

    dataset = NvidiaBertDataset(shard)
    assert len(dataset) == NUM_SAMPLES
    assert isinstance(dataset[0], list)


def test_load_dataset_from_shard(data_dir: pathlib.Path) -> None:
    shard = str(next(data_dir.iterdir()))

    loader = load_dataset_from_shard(shard, 1, num_replicas=1, rank=0)
    assert len(loader) == NUM_SAMPLES

    loader = load_dataset_from_shard(shard, 4, num_replicas=1, rank=0)
    assert len(loader) == NUM_SAMPLES // 4


def test_sharded_dataset(data_dir: pathlib.Path) -> None:
    batch_size = 4
    dataset = sharded_dataset(
        str(data_dir),
        batch_size,
        num_replicas=1,
        rank=0,
        num_workers=0,
    )

    batches = 0
    for batch in dataset:
        batches += 1
        assert isinstance(batch, Batch)

    assert batches == (NUM_FILE * NUM_SAMPLES // batch_size)
