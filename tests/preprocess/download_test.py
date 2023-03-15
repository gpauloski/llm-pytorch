from __future__ import annotations

from unittest import mock

import click.testing

from llm.preprocess.download import cli


def test_cli() -> None:
    # Testing llm.preprocess.download is challenging because it basically
    # downloads a lot of data.
    # TODO: make integration tests that run weekly and run the downloader
    runner = click.testing.CliRunner()

    with mock.patch('llm.preprocess.download.download_wikipedia'):
        result = runner.invoke(
            cli,
            ['--dataset', 'wikipedia', '--output', '/tmp/download'],
        )
        assert result.exit_code == 0

    with mock.patch('llm.preprocess.download.download_bookscorpus'):
        result = runner.invoke(
            cli,
            ['--dataset', 'bookscorpus', '--output', '/tmp/download'],
        )
        assert result.exit_code == 0

    result = runner.invoke(
        cli,
        ['--dataset', 'unknown', '--output', '/tmp/download'],
    )
    assert result.exit_code == 2
