from __future__ import annotations

import pytest
import torch

from llm.datasets.sharded import DatasetParams
from llm.datasets.sharded import DistributedShardedDataset
from llm.datasets.sharded import ResumableSequentialSampler


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, shard_offset: int) -> None:
        self.samples = samples
        self.shard_offset = shard_offset

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> int:
        assert 0 <= index < self.samples
        return self.shard_offset + index


def simple_dataset_params(
    shard_samples: list[int],
) -> dict[str, DatasetParams]:
    params: dict[str, DatasetParams] = {}
    offset = 0
    for i, samples in enumerate(shard_samples):
        kwargs = {'samples': samples, 'shard_offset': offset}
        params[f'shard-{i}'] = ((), kwargs)
        offset += samples
    return params


def test_rank_validity() -> None:
    with pytest.raises(ValueError):
        DistributedShardedDataset(SimpleDataset, {}, rank=-1, world_size=1)

    with pytest.raises(ValueError):
        DistributedShardedDataset(SimpleDataset, {}, rank=1, world_size=1)


def test_empty_dataset_params() -> None:
    with pytest.raises(ValueError):
        DistributedShardedDataset(SimpleDataset, {}, rank=0, world_size=1)


@pytest.mark.parametrize(
    'ranks,shard_samples,expected_samples_per_rank',
    (
        (1, [100], 100),
        (1, [100, 200], 300),
        (2, [100, 100], 100),
        (4, [100, 100], 50),
        (4, [100, 101], 50),
        (4, [100, 200, 300, 400], 250),
    ),
)
def test_local_size(
    ranks: int,
    shard_samples: list[int],
    expected_samples_per_rank: int,
) -> None:
    for rank in range(ranks):
        params = simple_dataset_params(shard_samples)
        dataset = DistributedShardedDataset(
            SimpleDataset,
            params,
            rank=rank,
            world_size=ranks,
        )
        assert len(dataset) == expected_samples_per_rank


def test_index_out_of_range() -> None:
    samples = 10
    params = simple_dataset_params([samples])
    dataset = DistributedShardedDataset(
        SimpleDataset,
        params,
        rank=0,
        world_size=1,
    )
    with pytest.raises(IndexError):
        # Max index is samples - 1
        dataset[10]


@pytest.mark.parametrize('shuffle', (True, False))
def test_iterating_single_rank(shuffle: bool) -> None:
    samples_per_shard = [10, 20, 30]
    params = simple_dataset_params(samples_per_shard)
    dataset = DistributedShardedDataset(
        SimpleDataset,
        params,
        rank=0,
        world_size=1,
        shuffle=shuffle,
    )
    samples = {dataset[i] for i in range(len(dataset))}
    assert len(samples) == sum(samples_per_shard)
    assert samples == set(range(sum(samples_per_shard)))


def test_iterating_multiple_ranks() -> None:
    samples_per_shard = [13, 17, 19, 23, 29]
    params = simple_dataset_params(samples_per_shard)
    ranks = 3
    samples: list[int] = []
    samples_per_rank: list[int] = []

    for rank in range(ranks):
        dataset = DistributedShardedDataset(
            SimpleDataset,
            params,
            rank=rank,
            world_size=ranks,
        )
        samples_per_rank.append(len(dataset))
        samples.extend(dataset[i] for i in range(len(dataset)))

    # Check all ranks saw same number of samples
    assert len(set(samples_per_rank)) == 1
    total_samples = sum(samples_per_shard)
    expected_samples = total_samples - (total_samples % ranks)
    assert set(samples) == set(range(expected_samples))


def test_sequential_sampler() -> None:
    dataset = range(100)
    sampler = ResumableSequentialSampler(dataset)
    samples = [i for i in sampler]
    assert samples == list(dataset)
    assert len(dataset) == 100


def test_sequential_sampler_resume() -> None:
    dataset = range(10)
    sampler = ResumableSequentialSampler(dataset, start_index=1)
    assert next(iter(sampler)) == 1
    assert len(sampler) == 9
