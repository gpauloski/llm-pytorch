from __future__ import annotations

from llm.trainers.bert.convert import main


def test_nothing() -> None:
    # For now, main is not tested.
    assert callable(main)
