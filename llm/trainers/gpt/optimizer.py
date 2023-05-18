from __future__ import annotations

import torch


def get_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ['bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p
                for n, p in model.named_parameters()
                if not any(_nd in n for _nd in no_decay)
            ],
            'weight_decay': weight_decay,
        },
        {
            'params': [
                p
                for n, p in model.named_parameters()
                if any(_nd in n for _nd in no_decay)
            ],
            'weight_decay': 0.0,
        },
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
    )
