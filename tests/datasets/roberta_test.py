from __future__ import annotations

import pathlib

import torch

from llm.datasets.roberta import RoBERTaDataset
from testing.datasets.roberta import write_roberta_shard

MAX_SEQ_LEN = 128


def test_roberta_dataset(tmp_path: pathlib.Path) -> None:
    filepath = tmp_path / 'input.hdf5'
    tokenizer = write_roberta_shard(filepath, MAX_SEQ_LEN)

    mask_token_id = tokenizer.token_to_id('[MASK]')
    mask_token_prob = 0.2

    dataset = RoBERTaDataset(
        filepath,
        mask_token_id=mask_token_id,
        mask_token_prob=mask_token_prob,
        vocab_size=tokenizer.get_vocab_size(),
    )
    assert len(dataset) > 0

    for i in range(10):
        sample = dataset[i]

        assert (
            len(sample.input_ids)
            == len(sample.attention_mask)
            == len(sample.token_type_ids)
            == len(sample.masked_labels)
        )
        assert sample.next_sentence_label is None

        masked = sample.input_ids == mask_token_id
        masked_count = torch.sum(masked)
        assert 0 <= masked_count <= MAX_SEQ_LEN / 2
