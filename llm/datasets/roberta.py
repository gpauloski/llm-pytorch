"""Custom RoBERTa dataset provider.

This is designed to work with data produced by the RoBERTa encoder
preprocessing script in `llm.preprocess.roberta`.
"""
from __future__ import annotations

import pathlib
from typing import cast

import h5py
import torch
from torch.utils.data import Dataset

from llm.datasets.bert import Sample


class RoBERTaDataset(Dataset[Sample]):
    def __init__(
        self,
        input_file: pathlib.Path | str,
        mask_token_id: int,
        mask_token_prob: float,
        vocab_size: int,
    ) -> None:
        self.input_file = input_file
        self.mask_token_id = mask_token_id
        self.mask_token_prob = mask_token_prob
        self.vocab_size = vocab_size

        self.loaded = False
        with h5py.File(self.input_file, 'r') as f:
            self.samples = len(f['input_ids'])

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Sample:
        if not self.loaded:
            self._lazy_load()

        input_ids = cast(torch.LongTensor, self.input_ids[index])
        attention_mask = cast(torch.LongTensor, self.attention_masks[index])
        special_tokens_mask = cast(
            torch.BoolTensor,
            self.special_tokens_masks[index],
        )
        token_type_ids = cast(
            torch.LongTensor,
            torch.zeros_like(attention_mask),
        )

        input_ids, masked_labels = bert_mask_sequence(
            input_ids,
            special_tokens_mask=special_tokens_mask,
            mask_token_id=self.mask_token_id,
            mask_token_prob=self.mask_token_prob,
            vocab_size=self.vocab_size,
        )

        return Sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_labels=masked_labels,
            next_sentence_label=None,
        )

    def _lazy_load(self) -> None:
        with h5py.File(self.input_file, 'r') as f:
            input_ids = f['input_ids'][:]
            attention_masks = f['attention_masks'][:]
            special_tokens_masks = f['special_tokens_masks'][:]

        self.input_ids = torch.from_numpy(input_ids).to(torch.long)
        self.attention_masks = torch.from_numpy(attention_masks).to(torch.long)
        self.special_tokens_masks = torch.from_numpy(
            special_tokens_masks,
        ).to(torch.bool)

        self.loaded = True


def bert_mask_sequence(
    token_ids: torch.LongTensor,
    special_tokens_mask: torch.BoolTensor,
    mask_token_id: int,
    mask_token_prob: float,
    vocab_size: int,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    # Source: https://github.com/huggingface/transformers/blob/f7329751fe5c43365751951502c00df5a4654359/src/transformers/data/data_collator.py#L748  # noqa: E501
    masked_labels = cast(torch.LongTensor, token_ids.clone())

    probability_matrix = torch.full(token_ids.shape, mask_token_prob)
    special_tokens_mask = special_tokens_mask.bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # Set non-masked tokens to -100 so loss is only computed on masked tokens
    masked_labels[~masked_indices] = -100

    # 80% of the time replace masked token with [MASK]
    indices_replaced = (
        torch.bernoulli(torch.full(token_ids.shape, 0.8)).bool()
        & masked_indices
    )
    token_ids[indices_replaced] = mask_token_id

    # 10% of the time replace masked tokens with random token
    indices_random = (
        torch.bernoulli(torch.full(token_ids.shape, 0.5)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(vocab_size, token_ids.shape, dtype=torch.long)
    token_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) the masked tokens are unchanged
    return token_ids, masked_labels
