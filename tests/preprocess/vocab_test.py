from __future__ import annotations

import pathlib
from typing import Generator
from typing import Literal

import pytest

from llm.preprocess.vocab import train_vocab
from testing.preprocess.text import random_document


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
        vocab = [token.strip() for token in f]

    if kind == 'bpe':
        assert len(vocab) >= 500
    elif kind == 'wordpiece':
        assert len(vocab) == 502
    else:
        raise AssertionError

    assert vocab[: len(special)] == special
