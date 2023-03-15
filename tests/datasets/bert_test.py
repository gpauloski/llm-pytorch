from __future__ import annotations

import pathlib
import tempfile
from collections.abc import Generator

import numpy as np
import pytest

from llm.datasets.bert import Batch
from llm.datasets.bert import NvidiaBertDataset
from llm.datasets.bert import Sample
from llm.datasets.bert import WorkerInitObj
from llm.datasets.bert import get_dataloader_from_nvidia_bert_shard
from llm.datasets.bert import get_masked_labels
from llm.datasets.bert import sharded_nvidia_bert_dataset
from testing.datasets.bert import write_nvidia_bert_shard

NUM_FILE = 3
NUM_SAMPLES = 32
SEQ_LEN = 128
VOCAB_SIZE = 30000


@pytest.fixture(scope='module')
def data_dir() -> Generator[pathlib.Path, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)

        for index in range(NUM_FILE):
            filepath = tmp_path / f'{index}.hdf5'
            write_nvidia_bert_shard(
                filepath,
                samples=NUM_SAMPLES,
                seq_len=SEQ_LEN,
                vocab_size=VOCAB_SIZE,
            )

        yield tmp_path


def test_worker_init() -> None:
    init = WorkerInitObj(0)
    init(3)


@pytest.mark.parametrize(
    'seq_len,masked_lm_positions,masked_lm_ids,result',
    (
        (1, [], [], [-1]),
        (2, [1], [42], [-1, 42]),
        (4, [1, 3], [42, 43], [-1, 42, -1, 43]),
        (
            14,
            [6, 12, 13],
            [1997, 2074, 2151],
            [-1, -1, -1, -1, -1, -1, 1997, -1, -1, -1, -1, -1, 2074, 2151],
        ),
    ),
)
def test_get_masked_labels(
    seq_len: int,
    masked_lm_positions: list[int],
    masked_lm_ids: list[int],
    result: list[int],
) -> None:
    labels = get_masked_labels(seq_len, masked_lm_positions, masked_lm_ids)
    assert np.array_equal(labels, result)


def test_dataset(data_dir: pathlib.Path) -> None:
    shard = str(next(data_dir.iterdir()))

    dataset = NvidiaBertDataset(shard)
    assert len(dataset) == NUM_SAMPLES
    assert isinstance(dataset[0], Sample)
    assert isinstance(dataset[1], Sample)


def test_get_dataloader_from_nvidia_bert_shard(data_dir: pathlib.Path) -> None:
    shard = str(next(data_dir.iterdir()))

    loader = get_dataloader_from_nvidia_bert_shard(
        shard,
        1,
        num_replicas=1,
        rank=0,
    )
    assert len(loader) == NUM_SAMPLES

    loader = get_dataloader_from_nvidia_bert_shard(
        shard,
        4,
        num_replicas=1,
        rank=0,
    )
    assert len(loader) == NUM_SAMPLES // 4


def test_sharded_nvidia_bert_dataset(data_dir: pathlib.Path) -> None:
    batch_size = 4
    dataset = sharded_nvidia_bert_dataset(
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
