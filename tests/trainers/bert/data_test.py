from __future__ import annotations

import pathlib
from unittest import mock

import torch

from llm.trainers.bert.data import get_dataloader
from llm.trainers.bert.data import get_dataset
from llm.trainers.bert.data import NvidiaBertDatasetConfig
from llm.trainers.bert.data import RobertaDatasetConfig
from testing.datasets.bert import write_nvidia_bert_shard


def test_load_nvidia_dataset_get_dataloader(tmp_path: pathlib.Path) -> None:
    write_nvidia_bert_shard(tmp_path / 'shard.h5', samples=11, seq_len=16)

    with (
        mock.patch('torch.distributed.get_rank', return_value=0),
        mock.patch('torch.distributed.get_world_size', return_value=1),
    ):
        dataset = get_dataset(NvidiaBertDatasetConfig(tmp_path))

    assert len(dataset) == 11
    dataloader = get_dataloader(
        dataset,
        torch.utils.data.SequentialSampler(dataset),
        batch_size=2,
    )
    assert len(dataloader) == 5


def test_load_roberta_dataset(tmp_path: pathlib.Path) -> None:
    # This shard only works with RobertaDatasetConfig because of its lazy
    # loading
    write_nvidia_bert_shard(tmp_path / 'shard.h5', samples=11, seq_len=16)

    class MockTokenizer:
        def get_vocab_size(self) -> int:
            return 100

        def token_to_id(self, token: str) -> int:
            return 100

    with (
        mock.patch('torch.distributed.get_rank', return_value=0),
        mock.patch('torch.distributed.get_world_size', return_value=1),
        mock.patch(
            'tokenizers.Tokenizer.from_file',
            return_value=MockTokenizer(),
        ),
    ):
        get_dataset(
            RobertaDatasetConfig(
                input_dir=tmp_path,
                tokenizer_file=tmp_path / 'tokenizer.json',
                mask_token_prob=0.15,
            ),
        )
