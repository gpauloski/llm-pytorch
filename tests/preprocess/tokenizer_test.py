from __future__ import annotations

import json
import pathlib
from typing import Generator
from typing import Literal
from unittest import mock

import click.testing
import pytest
import tokenizers

from llm.preprocess.tokenizer import cli
from llm.preprocess.tokenizer import get_tokenizer
from llm.preprocess.tokenizer import train_vocab
from testing.preprocess.text import random_document


@pytest.mark.parametrize('kind', ('bpe', 'wordpiece'))
def test_tokenizer(kind: Literal['bpe', 'wordpiece']) -> None:
    assert isinstance(
        get_tokenizer(kind),
        tokenizers.implementations.base_tokenizer.BaseTokenizer,
    )


@pytest.mark.parametrize('kind', ('bpe', 'wordpiece'))
def test_tokenizer_with_max_seq_len(kind: Literal['bpe', 'wordpiece']) -> None:
    assert isinstance(
        get_tokenizer(
            kind,
            padding_options={'length': 128},
            truncation_options={'max_length': 128},
        ),
        tokenizers.implementations.base_tokenizer.BaseTokenizer,
    )


@pytest.fixture
def input_files(
    tmp_path: pathlib.Path,
) -> Generator[list[pathlib.Path], None, None]:
    files = []
    for i in range(3):
        input_file = tmp_path / f'{i}.txt'
        with open(input_file, 'w') as f:
            for _k in range(5):
                f.write(random_document(sentences=7))
        files.append(input_file)

    yield files


@pytest.mark.parametrize(
    'kind,lowercase,special',
    (
        ('bpe', True, []),
        ('wordpiece', False, ['__PAD__', '__MASK__']),
    ),
)
def test_train_vocab(
    kind: Literal['bpe', 'wordpiece'],
    lowercase: bool,
    special: list[str],
    input_files: list[pathlib.Path],
    tmp_path: pathlib.Path,
) -> None:
    vocab_file = tmp_path / 'vocab.txt'
    train_vocab(
        kind,
        input_files=input_files,
        output_file=vocab_file,
        size=500,
        lowercase=lowercase,
        special_tokens=special,
    )

    with open(vocab_file) as f:
        vocab = json.load(f)['model']['vocab']

    if kind == 'bpe':
        assert len(vocab) >= 500
    elif kind == 'wordpiece':
        assert len(vocab) == 500
    else:
        raise AssertionError


def test_cli() -> None:
    runner = click.testing.CliRunner()
    with (
        mock.patch('llm.preprocess.tokenizer.train_vocab'),
        mock.patch('llm.preprocess.tokenizer.init_logging'),
    ):
        result = runner.invoke(cli, ['--input', 'x', '--output', 'y'])
        assert result.exit_code == 0
