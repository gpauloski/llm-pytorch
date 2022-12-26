from __future__ import annotations

import torch

from llm.loss import BertPretrainingCriterion


def test_bert_pretraining_criterion() -> None:
    vocab_size = 100
    batch_size = 4
    seq_len = 32

    prediction_scores = torch.rand(batch_size, seq_len, vocab_size)
    masked_lm_labels = torch.randint(
        0,
        vocab_size,
        (batch_size, seq_len),
        dtype=torch.int64,
    )

    seq_relationship_scores = torch.rand(batch_size, 2)
    next_sentence_labels = torch.randint(0, 2, (batch_size,))

    criterion = BertPretrainingCriterion(vocab_size)

    loss = criterion(prediction_scores, masked_lm_labels)
    loss_nsp = criterion(
        prediction_scores,
        masked_lm_labels,
        seq_relationship_scores,
        next_sentence_labels,
    )

    assert loss_nsp > loss > 0
