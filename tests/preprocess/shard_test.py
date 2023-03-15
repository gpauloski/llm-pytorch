from __future__ import annotations

import pathlib
import sys
from unittest import mock

import click.testing
import pytest

from llm.preprocess.shard import cli
from llm.preprocess.shard import get_document_shards
from llm.preprocess.shard import shard
from testing.preprocess.text import random_document


def test_get_document_shards() -> None:
    shards = list(get_document_shards([b'test', b'test'], 1000))
    assert shards == [[b'test', b'test']]

    bytes_per_shard = 2000
    documents = [random_document(10).encode('utf-8') for _ in range(10)]
    shards = list(get_document_shards(documents, bytes_per_shard))

    for shard_ in shards:
        assert sum(sys.getsizeof(doc) for doc in shard_) <= bytes_per_shard

    assert sum(len(shard) for shard in shards) == len(documents)


@pytest.mark.parametrize('shuffle', (True, False))
def test_shard(shuffle: bool, tmp_path: pathlib.Path) -> None:
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    input_files: list[pathlib.Path] = []

    # Write two input files with 5 documents each and 10 lines per document
    for i in range(2):
        input_file = input_dir / f'{i}.txt'
        with open(input_file, 'w') as f:
            for _ in range(5):
                f.write(random_document(10))
                f.write('\n\n')
        input_files.append(input_file)

    output_dir = tmp_path / 'output'

    shard(
        input_files,
        output_dir,
        'shard-{index}.txt',
        2000,
        shuffle,
    )

    shards = list(output_dir.glob('shard-*.txt'))
    assert len(shards) >= 1


def test_shard_bad_format(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError):
        shard([tmp_path], tmp_path, 'shard.txt', 100)


def test_cli() -> None:
    runner = click.testing.CliRunner()
    with (
        mock.patch('llm.preprocess.shard.shard'),
        mock.patch('llm.preprocess.shard.init_logging'),
    ):
        result = runner.invoke(
            cli,
            ['--input', 'x', '--output', 'y', '--size', '30 MB'],
        )
        assert result.exit_code == 0
