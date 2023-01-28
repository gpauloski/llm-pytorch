"""NVIDIA BERT dataset provider.

Source: https://github.com/hpcaitech/ColossalAI-Examples/blob/e0830ccc1bbc57f9c50bb1c00f3e23239bf1e231/language/roberta/pretraining/nvidia_bert_dataset_provider.py

TODO:
  - shard prefetching in separate thread
  - resume from shard
  - saving shard sampler state for state dict
"""  # noqa: D405, E501
from __future__ import annotations

import os
import random
from collections.abc import Generator
from typing import NamedTuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


class Batch(NamedTuple):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    masked_labels: torch.LongTensor
    next_sentence_labels: torch.LongTensor
    segment_ids: torch.LongTensor


# Workaround because python functions are not picklable
class WorkerInitObj:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, id_: int) -> None:
        np.random.seed(seed=self.seed + id_)
        random.seed(self.seed + id_)


class NvidiaBertDataset(Dataset):
    def __init__(self, input_file: str) -> None:
        with h5py.File(input_file, 'r') as f:
            self.input_ids = f['input_ids'][:]
            self.input_mask = f['input_mask'][:]
            self.masked_lm_ids = f['masked_lm_ids'][:]
            self.masked_lm_positions = f['masked_lm_positions'][:]
            self.next_sentence_labels = f['next_sentence_labels'][:]
            self.segment_ids = f['segment_ids'][:]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> list[torch.Tensor]:
        sample = [
            self.input_ids[index].astype(np.int64),
            self.input_mask[index].astype(np.int64),
            masked_labels(
                len(self.input_ids[index]),
                self.masked_lm_positions[index],
                self.masked_lm_ids[index],
            ),
            np.asarray(self.next_sentence_labels[index]).astype(np.int64),
            self.segment_ids[index].astype(np.int64),
        ]

        return [torch.from_numpy(x) for x in sample]


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
    masked_lm_labels[masked_lm_positions] = masked_lm_ids
    return masked_lm_labels


def get_shard_filepaths(input_dir: str) -> list[str]:
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith('.h5') or f.endswith('.hdf5')
    ]


def load_dataset_from_shard(
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


def sharded_dataset(
    input_dir: str,
    batch_size: int,
    *,
    num_replicas: int | None = None,
    rank: int | None = None,
    seed: int = 0,
    num_workers: int = 4,
) -> Generator[Batch, None, None]:
    shard_filepaths = get_shard_filepaths(input_dir)
    shard_filepaths.sort()

    for shard_filepath in shard_filepaths:
        dataloader = load_dataset_from_shard(
            shard_filepath,
            batch_size,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
        )
        for batch in dataloader:
            yield Batch(*batch)
