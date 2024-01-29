"""Utilities for loading BERT models from HuggingFace.

Source: [ColossalAI-Examples](https://github.com/hpcaitech/ColossalAI-Examples/blob/e0830ccc1bbc57f9c50bb1c00f3e23239bf1e231/language/bert/colotensor/model/__init__.py){target=_blank}
"""

from __future__ import annotations

from typing import Any

import transformers

BERT_BASE = dict(
    attention_probs_dropout_prob=0.1,
    hidden_act='gelu_new',
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=2,
    vocab_size=30522,
)
"""BERT-base HuggingFace configuration."""

BERT_LARGE = dict(
    attention_probs_dropout_prob=0.1,
    hidden_act='gelu_new',
    hidden_dropout_prob=0.1,
    hidden_size=1024,
    initializer_range=0.02,
    intermediate_size=4096,
    max_position_embeddings=512,
    num_attention_heads=16,
    num_hidden_layers=24,
    type_vocab_size=2,
    vocab_size=30522,
)
"""BERT-large HuggingFace configuration."""


def from_config(
    config: dict[str, Any],
    checkpoint_gradients: bool = False,
) -> transformers.BertForPreTraining:
    """Load a BERT model from the configuration.

    Args:
        config: BERT configuration.
        checkpoint_gradients: Enable gradient checkpointing.

    Returns:
        BERT model.
    """
    config = transformers.BertConfig(**config)
    model = transformers.BertForPreTraining(config)
    if checkpoint_gradients:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    return model
