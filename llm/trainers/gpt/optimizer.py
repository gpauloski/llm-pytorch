from __future__ import annotations

import logging
from typing import Any
from typing import Callable

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


def get_preconditioner(
    model: torch.nn.Module,
    factor_update_steps: Callable[[int], int] | int,
    inv_update_steps: Callable[[int], int] | int,
    learning_rate: Callable[[int], float] | float,
    accumulation_steps: int,
    **kwargs: Any,
) -> Any:
    # Note: return Any here and import inside the function in case someone
    # is not using KFAC and does not have it installed
    from kfac.enums import AssignmentStrategy
    from kfac.enums import DistributedStrategy
    from kfac.preconditioner import KFACPreconditioner

    preconditioner = KFACPreconditioner(
        model,
        factor_update_steps=factor_update_steps,
        inv_update_steps=inv_update_steps,
        lr=learning_rate,
        accumulation_steps=accumulation_steps,
        # GPT is big so the following options optimize for memory
        assignment_strategy=AssignmentStrategy.MEMORY,
        compute_eigenvalue_outer_product=False,
        grad_worker_fraction=DistributedStrategy.MEM_OPT,
        loglevel=(
            logging.INFO
            if torch.distributed.get_rank() == 0
            else logging.DEBUG
        ),
        **kwargs,
    )

    return preconditioner
