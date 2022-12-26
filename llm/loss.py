from __future__ import annotations

import torch


class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = vocab_size

    def forward(
        self,
        prediction_scores: torch.FloatTensor,
        masked_lm_labels: torch.LongTensor,
        seq_relationship_score: torch.FloatTensor = None,
        next_sentence_labels: torch.LongTensor = None,
    ) -> float:
        masked_lm_loss = self.loss_fn(
            prediction_scores.view(-1, self.vocab_size),
            masked_lm_labels.view(-1),
        )

        if (
            seq_relationship_score is not None
            and next_sentence_labels is not None
        ):
            next_sentence_loss = self.loss_fn(
                seq_relationship_score.view(-1, 2),
                next_sentence_labels.view(-1),
            )
            masked_lm_loss += next_sentence_loss

        return masked_lm_loss
