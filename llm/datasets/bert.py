"""NVIDIA BERT dataset provider.

Source: https://github.com/hpcaitech/ColossalAI-Examples/blob/e0830ccc1bbc57f9c50bb1c00f3e23239bf1e231/language/roberta/pretraining/nvidia_bert_dataset_provider.py
"""

from __future__ import annotations

import random
from collections.abc import Generator
from typing import Any
from typing import cast
from typing import NamedTuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from llm.utils import get_filepaths


class Batch(NamedTuple):
    """BERT pretraining batch."""

    input_ids: torch.LongTensor
    """Input sequence token IDs (`(batch_size, seq_len)`)."""
    attention_mask: torch.LongTensor
    """Input sequence attention mask (`(batch_size, seq_len)`).

    Indicates which tokens in `input_ids` should be attended to (i.e., are not
    padding tokens). Also known as the input mask.
    """
    token_type_ids: torch.LongTensor
    """Token IDs indicating the segment labels (`(batch_size, seq_len)`).

    E.g. if `input_ids` is composed of two distinct segments, the first segment
    will have token IDs set to 0 and the second to 1. Also known as the
    segment ids.
    """
    masked_labels: torch.LongTensor
    """True token ID of masked tokens in `input_ids` (`(batch_size, seq_len)`).

    Indices corresponding to non-masked tokens in `input_ids` are typically
    set to `-100` to avoid contributing to the MLM loss.
    """
    next_sentence_labels: torch.LongTensor | None
    """Boolean tensor indicating the next sentence label (`(batch_size,)`).

    A true (`1`) value indicates the next sentence/segment logically follows
    the first. If there is only one segment in `input_ids`, this can be
    set to `None`.
    """


class Sample(NamedTuple):
    input_ids: torch.LongTensor
    """Input sequence token IDs (`(seq_len,)`)."""
    attention_mask: torch.LongTensor
    """Input sequence attention mask (`(seq_len,)`).

    Indicates which tokens in `input_ids` should be attended to (i.e., are not
    padding tokens). Also known as the input mask.
    """
    token_type_ids: torch.LongTensor
    """Token IDs indicating the segment labels (`(seq_len,)`).

    E.g. if `input_ids` is composed of two distinct segments, the first segment
    will have token IDs set to 0 and the second to 1. Also known as the
    segment ids.
    """
    masked_labels: torch.LongTensor
    """True token ID of masked tokens in `input_ids` (`(seq_len,)`).

    Indices corresponding to non-masked tokens in `input_ids` are typically
    set to `-100` to avoid contributing to the MLM loss.
    """
    next_sentence_label: torch.LongTensor | None
    """Boolean scalar tensor indicating the next sentence label.

    A true (`1`) value indicates the next sentence/segment logically follows
    the first. If there is only one segment in `input_ids`, this can be
    set to `None`.
    """


# Workaround because python functions are not picklable
class WorkerInitObj:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, id_: int) -> None:
        np.random.seed(seed=self.seed + id_)
        random.seed(self.seed + id_)


class NvidiaBertDataset(Dataset[Sample]):
    """NVIDIA BERT dataset.

    Like the PyTorch [`Dataset`][torch.utils.data.Dataset], this dataset is
    indexable returning a [`Sample`][llm.datasets.bert.Sample].

    Example:
        ```python
        >>> from llm.datasets.bert import NvidiaBertDataset
        >>> dataset = NvidiaBertDataset('/path/to/shard')
        >>> dataset[5]
        Sample(...)
        ```

    Args:
        input_file: HDF5 file to load.
    """

    def __init__(self, input_file: str) -> None:
        self.input_file = input_file

        # We just inspect the file enough to find the size of the dataset
        # then defer loading until we actually use the dataset. This is
        # particularly helpful for the DistributedShardedDataset which needs
        # to get the length of each shard
        self.loaded = False
        with h5py.File(self.input_file, 'r') as f:
            self.samples = len(f['next_sentence_labels'])

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if not self.loaded:
            self._lazy_load()

        input_ids = self.input_ids[index].astype(np.int64)
        attention_mask = self.input_mask[index].astype(np.int64)
        token_type_ids = self.segment_ids[index].astype(np.int64)
        masked_labels = get_masked_labels(
            len(self.input_ids[index]),
            self.masked_lm_positions[index],
            self.masked_lm_ids[index],
        )
        next_sentence_label = np.asarray(
            self.next_sentence_labels[index],
        ).astype(np.int64)

        input_ids_ = cast(torch.LongTensor, torch.from_numpy(input_ids))
        attention_mask_ = cast(
            torch.LongTensor,
            torch.from_numpy(attention_mask),
        )
        token_type_ids_ = cast(
            torch.LongTensor,
            torch.from_numpy(token_type_ids),
        )
        masked_labels_ = cast(
            torch.LongTensor,
            torch.from_numpy(masked_labels),
        )
        next_sentence_label_ = cast(
            torch.LongTensor,
            torch.from_numpy(next_sentence_label),
        )

        return Sample(
            input_ids=input_ids_,
            attention_mask=attention_mask_,
            token_type_ids=token_type_ids_,
            masked_labels=masked_labels_,
            next_sentence_label=next_sentence_label_,
        )

    def _lazy_load(self) -> None:
        with h5py.File(self.input_file, 'r') as f:
            self.input_ids = f['input_ids'][:]
            self.input_mask = f['input_mask'][:]
            self.segment_ids = f['segment_ids'][:]
            self.masked_lm_ids = f['masked_lm_ids'][:]
            self.masked_lm_positions = f['masked_lm_positions'][:]
            self.next_sentence_labels = f['next_sentence_labels'][:]
        self.loaded = True


def get_masked_labels(
    seq_len: int,
    masked_lm_positions: list[int],
    masked_lm_ids: list[int],
) -> np.ndarray[Any, Any]:
    """Create masked labels array.

    Args:
        seq_len: Sequence length.
        masked_lm_positions: Index in sequence of masked tokens
        masked_lm_ids: True token value for each position in
            `masked_lm_position`.

    Returns:
        List with length `seq_len` with the true value for each corresponding \
        masked token in `input_ids` and -100 for all tokens in input_ids \
        which are not masked.
    """
    masked_lm_labels = np.ones(seq_len, dtype=np.int64) * -100
    if len(masked_lm_positions) > 0:
        # store number of  masked tokens in index
        (padded_mask_indices,) = np.nonzero(np.array(masked_lm_positions) == 0)
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0]
        else:
            index = len(masked_lm_positions)
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
    return masked_lm_labels


def get_dataloader_from_nvidia_bert_shard(
    input_file: str,
    batch_size: int,
    *,
    num_replicas: int | None = None,
    rank: int | None = None,
    seed: int = 0,
    num_workers: int = 4,
) -> DataLoader[Batch]:
    """Create a dataloader from a dataset shard.

    Args:
        input_file: HDF5 file to load.
        batch_size: Size of batches yielded by the dataloader.
        num_replicas: Number of processes participating in distributed
            training.
        rank: Rank of the current process within `num_replicas`.
        seed: Random seed used to shuffle the sampler.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        Dataloader which can be iterated over to yield \
        [`Batch`][llm.datasets.bert.Batch].
    """
    dataset = NvidiaBertDataset(input_file)
    sampler: DistributedSampler[int] = DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=True,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=4,
        worker_init_fn=WorkerInitObj(seed),
        pin_memory=True,
    )

    return dataloader


def sharded_nvidia_bert_dataset(
    input_dir: str,
    batch_size: int,
    *,
    num_replicas: int | None = None,
    rank: int | None = None,
    seed: int = 0,
    num_workers: int = 4,
) -> Generator[Batch, None, None]:
    """Simple generator which yields pretraining batches.

    Args:
        input_dir: Directory of HDF5 shards to load samples from.
        batch_size: Size of batches yielded.
        num_replicas: Number of processes participating in distributed
            training.
        rank: Rank of the current process within `num_replicas`.
        seed: Random seed used to shuffle the sampler.
        num_workers: Number of subprocesses to use for data loading.

    Yields:
        Batches of pretraining data.
    """
    shard_filepaths = get_filepaths(
        input_dir,
        extensions=['.h5', '.hdf5'],
        recursive=True,
    )
    shard_filepaths.sort()

    for shard_filepath in shard_filepaths:
        dataloader = get_dataloader_from_nvidia_bert_shard(
            shard_filepath,
            batch_size,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )
        for batch in dataloader:
            yield Batch(*batch)
