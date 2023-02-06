from __future__ import annotations

from llm.trainers.bert.main import main


def test_nothing() -> None:
    # For now, main is not tested.
    assert callable(main)
