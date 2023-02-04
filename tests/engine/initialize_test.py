from __future__ import annotations

import torch

from llm.engine.base import BaseOptimizer
from llm.engine.initialize import initialize


def test_initalize_default() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    _, optimizer, _, _ = initialize(model, optimizer, criterion, scheduler)

    assert isinstance(optimizer, BaseOptimizer)


def test_initialize_full() -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    initialize(
        model,
        optimizer,
        criterion,
        scheduler,
        accumulation_steps=2,
        dtype=torch.bfloat16,
        max_norm=0.1,
        init_scale=2**15,
    )
