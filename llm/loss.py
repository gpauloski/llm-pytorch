"""Training loss functions."""
from __future__ import annotations

import torch


class BertPretrainingCriterion(torch.nn.Module):
    """BERT pretraining loss.

    Computes the sum of the cross entropy losses of the masked language model
    and (optionally) next sentence prediction tasks.

    Args:
        vocab_size: Size of the pretraining vocabulary.
        ignore_index: Value to ignore when computing cross entropy loss.
            Defaults to -100 which is used by the provided BERT datasets
            as the value in `masked_lm_labels` which are not masked.
    """

    def __init__(self, vocab_size: int, ignore_index: int = -100) -> None:
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.vocab_size = vocab_size

    def forward(
        self,
        prediction_scores: torch.FloatTensor,
        masked_lm_labels: torch.LongTensor,
        seq_relationship_score: torch.FloatTensor | None = None,
        next_sentence_labels: torch.LongTensor | None = None,
    ) -> float:
        """Compute the pretraining loss.

        Args:
            prediction_scores: Masked token predictions.
            masked_lm_labels: True masked token labels.
            seq_relationship_score: Predicted sequence relationship score.
            next_sentence_labels: True next sentence label.

        Returns:
            Computed loss.
        """
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
