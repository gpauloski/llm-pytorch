from __future__ import annotations

import torch

from llm.engine.base import BaseOptimizer


def test_base_optimizer() -> None:
    model = torch.nn.Linear(1, 1)
    old = torch.optim.SGD(model.parameters(), lr=0.1)
    new = BaseOptimizer(old)

    assert old.param_groups == new.param_groups
    assert old.defaults == new.defaults

    state_dict = new.state_dict()
    state_dict['param_groups'][0]['lr'] = 0.2
    new.load_state_dict(state_dict)
    assert old.state_dict()['param_groups'][0]['lr'] == 0.2
