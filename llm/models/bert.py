"""Utilities for loading BERT models from HuggingFace.

Source: https://github.com/hpcaitech/ColossalAI-Examples/blob/e0830ccc1bbc57f9c50bb1c00f3e23239bf1e231/language/bert/colotensor/model/__init__.py
"""  # noqa: E501
from __future__ import annotations

from typing import Any

from colossalai.core import global_context as gpc
from transformers import BertConfig
from transformers import BertForPreTraining

BERT_BASE = dict(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
)

BERT_LARGE = dict(
    vocab_size=30522,
    hidden_size=1024,
    num_hidden_layers=24,
    num_attention_heads=16,
    intermediate_size=4096,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=512,
    type_vocab_size=2,
    initializer_range=0.02,
)


def from_config(config: dict[str, Any] | None = None) -> BertForPreTraining:
    if config is None:
        config = gpc.config.bert_config

    bert_config = BertConfig(**config)

    return BertForPreTraining(bert_config)
