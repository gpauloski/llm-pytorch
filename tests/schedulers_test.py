from __future__ import annotations

import torch

from llm.schedulers import LinearWarmupLR


def test_linear_warmup_schedule() -> None:
    base_lr = 0.1
    warmup_steps = 5
    total_steps = 20
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=base_lr)
    criterion = torch.nn.MSELoss()
    scheduler = LinearWarmupLR(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
    )

    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.zero_grad()
        batch = torch.rand(10, 1)
        target = 2 * batch

        output = module(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        scheduler.step()

    assert lrs[5] == 0.1
    assert lrs[0] < lrs[5]
    assert lrs[-1] < lrs[5]
