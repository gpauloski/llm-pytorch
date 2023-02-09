from __future__ import annotations

import random
from typing import Any
from typing import Generic
from typing import Mapping
from typing import TypeVar

import torch

DatasetType = TypeVar('DatasetType', bound=torch.utils.data.Dataset)
DatasetParams = tuple[tuple[Any, ...], Mapping[str, Any]]


class DistributedShardedDataset(
    torch.utils.data.Dataset,
    Generic[DatasetType],
):
    """Dataset wrapper for sharded datasets in distributed environments.

    The DistributedShardedDataset manages a set of datasets (shards) and
    restricts ranks to viewing a subset of the global indices across the
    shards. This is achieved by sorting the shards and counting the samples
    in each shard to compute the total number of samples then chunking those
    samples by rank.

    For example, if there are four ranks and eight shards of equal size, rank
    zero will see shards zero and one, rank two will see shards two and three,
    and so on. The length of the DistributedShardedDataset as seen by a rank
    will be (1 / world_size) * sum_of_samples_across_shards.

    The DistributedShardedDataset ensures only one shard is loaded at a time
    on a rank so the full dataset is never loaded into memory at once.

    Note:
        When building a DataLoader from a DistributedShardedDataset, do NOT
        use PyTorch's DistributedSampler. If you want to be able to save the
        state of the DataLoader, use the SequentialSampler because this class
        already provides the support for partitioning samples across ranks.

    Note:
        Samples at the end of the last shard will be dropped to ensure
        each rank sees an equal number of samples.

    Todo:
        * Next shard prefetching
        * Sample index shuffling within a shard
        * Support shuffle shard order by epoch

    Args:
        dataset_type: Dataset type that represents a single shard. This
            subtype of Dataset must be a map-style dataset. Iterable-style
            datasets are not supported.
        shard_params: dictionary mapping shard keys to the parameters used
            to initialize a ``dataset_type`` for the shard. The parameter type
            is a tuple of args and kwargs.
        rank: rank of this process.
        world_size: number of ranks sharing the dataset.
        shuffle: shuffle the shard order by the shard keys. The default
            (``False``) sorts the shards by shard key.
        seed: seed used for shuffling the shard order.
    """

    def __init__(
        self,
        dataset_type: type[DatasetType],
        shard_params: dict[str, DatasetParams],
        *,
        rank: int,
        world_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if not (0 <= rank < world_size):
            raise ValueError(
                f'Got rank={rank} which does not satisfy 0 <= rank < '
                f'world_size where world_size={world_size}.',
            )
        if len(shard_params) == 0:
            raise ValueError(
                'Parameters for at least one shard must be provided.',
            )

        random.seed(seed)

        self.dataset_type = dataset_type
        self.shard_params = shard_params
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle

        shard_keys = sorted(shard_params.keys())
        if shuffle:
            random.shuffle(shard_keys)

        # Mapping of shard_key to (start_index, end_index)
        shard_indices: dict[str, tuple[int, int]] = {}
        index = 0
        for shard_key in shard_keys:
            shard = self.load_shard(shard_key)
            shard_indices[shard_key] = (index, index + len(shard))
            index += len(shard)
            del shard

        # Drop indices from last shard to make divisible by world size
        last_shard_key = shard_keys[-1]
        last_shard_indices = shard_indices[last_shard_key]
        shard_indices[last_shard_key] = (
            last_shard_indices[0],
            last_shard_indices[1] - (last_shard_indices[1] % world_size),
        )

        self.shard_keys = shard_keys
        self.shard_indices = shard_indices
        self.total_samples = shard_indices[last_shard_key][1]

        assert len(shard_keys) == len(shard_indices) == len(shard_params)
        assert len(self) * self.world_size == self.total_samples

        self._current_shard_key: str | None = None
        self._current_shard: DatasetType | None = None

    def __len__(self) -> int:
        return self.total_samples // self.world_size

    def __getitem__(self, rank_index: int) -> Any:
        if rank_index >= len(self):
            raise IndexError(
                f'Requested sample index {rank_index} exceeds dataset size of'
                f'{len(self)} samples.',
            )
        shard_key, shard_index = self.rank_index_to_shard_index(rank_index)

        if (
            self._current_shard_key is None
            or self._current_shard_key != shard_key
        ):
            self._current_shard_key = shard_key
            self._current_shard = self.load_shard(shard_key)

        # If self._current_shard_key is not None then self._current_shard
        # should never be None.
        assert self._current_shard is not None

        return self._current_shard[shard_index]

    def rank_index_to_global_index(self, rank_index: int) -> int:
        rank_start_index = len(self) * self.rank
        return rank_start_index + rank_index

    def rank_index_to_shard_index(self, rank_index: int) -> tuple[str, int]:
        global_index = self.rank_index_to_global_index(rank_index)
        for shard_key in self.shard_keys:
            shard_indices = self.shard_indices[shard_key]
            if shard_indices[0] <= global_index < shard_indices[1]:
                return (shard_key, global_index - shard_indices[0])
        raise AssertionError(
            f'Rank index {rank_index} for rank {self.rank} maps to global '
            f'index {global_index} which exceeds the total samples in the '
            f'dataset ({self.total_samples}).',
        )

    def load_shard(self, shard_key: str) -> DatasetType:
        args, kwargs = self.shard_params[shard_key]
        return self.dataset_type(*args, **kwargs)
