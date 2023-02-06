from __future__ import annotations

import torch

from llm.engine import amp


def test_amp_training() -> None:
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()

    module, optimizer, criterion = amp.initialize(
        module,
        optimizer,
        criterion,
        # float16 casting not supported on CPU yet
        dtype=torch.bfloat16,
    )

    for _ in range(10):
        batch = torch.rand(10, 1)
        target = 2 * batch

        output = module(batch)
        loss = criterion(output, target)
        optimizer.backward(loss)
        optimizer.step()
        optimizer.zero_grad()


def test_amp_optimizer_state_dict() -> None:
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    optimizer = amp.AMPOptimizer(module, optimizer, scaler)
    state_dict = optimizer.state_dict()
    optimizer.load_state_dict(state_dict)


def test_amp_optimizer_grad_clip() -> None:
    module = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    criterion = torch.nn.MSELoss()

    optimizer = amp.AMPOptimizer(module, optimizer, scaler, max_norm=0.001)

    batch = torch.rand(10, 1)
    target = 2 * batch
    output = module(batch)
    loss = criterion(output, target)
    optimizer.backward(loss)
    optimizer.step()

    assert torch.norm(module.weight.grad, p=2.0) <= 0.001
