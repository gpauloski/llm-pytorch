from __future__ import annotations

from llm.trainers.bert import main


def test_nothing() -> None:
    # Nothing test for coverage (currently all the code in the
    # llm.trainers.bert module is no-covered).
    assert callable(main)
