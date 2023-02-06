from __future__ import annotations

import torch

from llm.engine import accumulation
from llm.engine.base import BaseOptimizer


def test_accumulation_training() -> None:
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
        optimizer,
        lambda x: 0.9,
    )
    criterion = torch.nn.MSELoss()

    optimizer, scheduler = accumulation.initialize(
        module,
        BaseOptimizer(optimizer),
        scheduler,
        accumulation_steps=10,
    )

    lrs = []

    for i in range(99):
        assert (i % 10 == 0) == optimizer.accumulation_boundary()

        batch = torch.rand(10, 1)
        target = 2 * batch

        output = module(batch)
        loss = criterion(output, target)

        optimizer.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert len(set(lrs)) == 10
