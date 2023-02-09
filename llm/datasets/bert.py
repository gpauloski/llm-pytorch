"""NVIDIA BERT dataset provider.

Source: https://github.com/hpcaitech/ColossalAI-Examples/blob/e0830ccc1bbc57f9c50bb1c00f3e23239bf1e231/language/roberta/pretraining/nvidia_bert_dataset_provider.py

TODO:
  - shard prefetching in separate thread
  - resume from shard
  - saving shard sampler state for state dict
"""  # noqa: D405, E501
from __future__ import annotations

import random
from collections.abc import Generator
from typing import NamedTuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from llm.utils import get_filepaths


class Batch(NamedTuple):
    # (batch_size, seq_len): indices of input sequence token in vocab
    input_ids: torch.LongTensor
    # (batch_size, seq_len): indicates which tokens in input_ids should be
    # attended to. Also know as the input_mask
    attention_mask: torch.LongTensor
    # (batch_size, seq_len): indicates first and second segments in input_ids.
    # Also known as segment_ids
    token_type_ids: torch.LongTensor
    # (batch_size, seq_len): indices in vocab of masked tokens in input_ids
    masked_labels: torch.LongTensor
    # (batch_size, ): boolean indicating next sentence label
    next_sentence_labels: torch.LongTensor


class Sample(NamedTuple):
    # (seq_len, ): indices of input sequence token in vocab
    input_ids: torch.LongTensor
    # (seq_len, ): indicates which tokens in input_ids should be
    # attended to. Also know as the input_mask
    attention_mask: torch.LongTensor
    # (seq_len, ): indicates first and second segments in input_ids.
    # Also known as segment_ids
    token_type_ids: torch.LongTensor
    # (seq_len, ): indices in vocab of masked tokens in input_ids
    masked_labels: torch.LongTensor
    # (1, ): boolean indicating next sentence label
    next_sentence_label: torch.LongTensor


# Workaround because python functions are not picklable
class WorkerInitObj:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, id_: int) -> None:
        np.random.seed(seed=self.seed + id_)
        random.seed(self.seed + id_)


class NvidiaBertDataset(Dataset):
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
        masked_labels_ = masked_labels(
            len(self.input_ids[index]),
            self.masked_lm_positions[index],
            self.masked_lm_ids[index],
        )
        next_sentence_label = np.asarray(
            self.next_sentence_labels[index],
        ).astype(np.int64)

        return Sample(
            input_ids=torch.from_numpy(input_ids),
            attention_mask=torch.from_numpy(attention_mask),
            token_type_ids=torch.from_numpy(token_type_ids),
            masked_labels=torch.from_numpy(masked_labels_),
            next_sentence_label=torch.from_numpy(next_sentence_label),
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


def masked_labels(
    seq_len: int,
    masked_lm_positions: list[int],
    masked_lm_ids: list[int],
) -> np.ndarray:
    """Create masked labels array.

    Args:
        seq_len (int): sequence length.
        masked_lm_positions (list[int]): index in sequence of masked tokens
        masked_lm_ids (list[int]): true token value for each position in
            masked_lm_position.

    Returns:
        List with length seq_len with the true value for each corresponding
        masked token in input_ids and -1 for all tokens in input_ids which are
        not masked.
    """
    masked_lm_labels = np.ones(seq_len, dtype=np.int64) * -1
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
) -> DataLoader:
    dataset = NvidiaBertDataset(input_file)
    sampler = DistributedSampler(
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
