from __future__ import annotations

from llm.preprocess.download import main


def test_nothing() -> None:
    # Testing llm.preprocess.download is challenging because it basically
    # downloads a lot of data.
    # TODO: make integration tests that run weekly and run the downloader
    assert callable(main)
