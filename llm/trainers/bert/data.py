from __future__ import annotations

import pathlib
from typing import NamedTuple

import tokenizers
import torch

from llm.datasets.bert import NvidiaBertDataset
from llm.datasets.bert import Sample
from llm.datasets.roberta import RoBERTaDataset
from llm.datasets.sharded import DatasetParams
from llm.datasets.sharded import DistributedShardedDataset
from llm.utils import get_filepaths


class NvidiaBertDatasetConfig(NamedTuple):
    """NVIDIA BERT pretraining dataset configuration."""

    input_dir: pathlib.Path | str


class RobertaDatasetConfig(NamedTuple):
    """RoBERTa pretraining dataset configuration."""

    input_dir: pathlib.Path | str
    tokenizer_file: pathlib.Path | str
    mask_token_prob: float


def get_dataloader(
    dataset: DistributedShardedDataset[Sample],
    sampler: torch.utils.data.Sampler[int],
    batch_size: int,
) -> torch.utils.data.DataLoader[Sample]:
    """Create a dataloader from a sharded dataset."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


def get_dataset(
    config: NvidiaBertDatasetConfig | RobertaDatasetConfig,
) -> DistributedShardedDataset[Sample]:
    """Load a sharded BERT pretraining dataset."""
    files = get_filepaths(
        config.input_dir,
        extensions=['.h5', '.hdf5'],
        recursive=True,
    )

    dataset = torch.utils.data.Dataset
    params: dict[str, DatasetParams]
    if isinstance(config, NvidiaBertDatasetConfig):
        dataset = NvidiaBertDataset
        params = {file: ((file,), {}) for file in files}
    elif isinstance(config, RobertaDatasetConfig):
        tokenizer = tokenizers.Tokenizer.from_file(config.tokenizer_file)

        dataset = RoBERTaDataset
        kwargs = {
            'mask_token_id': tokenizer.token_to_id('[MASK]'),
            'mask_token_prob': config.mask_token_prob,
            'vocab_size': tokenizer.get_vocab_size(),
        }
        params = {file: ((file,), kwargs) for file in files}
    else:
        raise AssertionError('Unreachable.')

    return DistributedShardedDataset(
        dataset,
        params,
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
    )
