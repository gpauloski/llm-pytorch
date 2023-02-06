from __future__ import annotations

import pytest
from transformers import BertForPreTraining

from llm.models.bert import from_config

TINY_CONFIG = dict(
    vocab_size=1000,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=64,
    type_vocab_size=2,
    initializer_range=0.02,
)


@pytest.mark.parametrize('checkpointing', (True, False))
def test_from_passed_config(checkpointing: bool) -> None:
    model = from_config(config=TINY_CONFIG, checkpoint_gradients=checkpointing)
    assert isinstance(model, BertForPreTraining)
