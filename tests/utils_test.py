from __future__ import annotations

import pytest

from llm.utils import gradient_accumulation_steps


def test_grad_accumulation_steps() -> None:
    assert gradient_accumulation_steps(8, 4, 2) == 1
    assert gradient_accumulation_steps(16, 4, 2) == 2
    assert gradient_accumulation_steps(32, 1, 1) == 32

    with pytest.raises(ValueError):
        gradient_accumulation_steps(8, 3, 2)
