from __future__ import annotations

from unittest import mock

import pytest
import torch

from llm.engine.base import BaseOptimizer
from llm.engine.initialize import initialize


@pytest.mark.parametrize(('cuda', 'dist'), ((True, False), (False, True)))
def test_initalize_default(cuda: bool, dist: bool) -> None:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    with (
        mock.patch('torch.cuda.is_available', return_value=cuda),
        mock.patch('torch.cuda.current_device', return_value=0),
        mock.patch('torch.distributed.is_initialized', return_value=dist),
        mock.patch('torch.nn.parallel.DistributedDataParallel'),
        mock.patch.object(model, 'cuda'),
    ):
        _, optimizer_, _, _ = initialize(
            model,
            optimizer,
            criterion,
            scheduler,
        )

    assert isinstance(optimizer_, BaseOptimizer)


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
