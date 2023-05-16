from __future__ import annotations

import pathlib

import pytest

from llm.preprocess.format import combine_document_files
from llm.preprocess.format import get_sent_tokenizer
from llm.preprocess.format import read_documents_bytes
from llm.preprocess.format import write_documents


def test_combine_document_files(tmp_path: pathlib.Path) -> None:
    docs = [tmp_path / f'doc{i}.txt' for i in range(3)]

    text = """\
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco.
laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.
Dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia.
Deserunt mollit anim id est laborum.
"""

    for doc in docs:
        with open(doc, 'w') as f:
            f.write(text)

    out = tmp_path / 'out.txt'
    combine_document_files(docs, out)

    with open(out) as f:
        out_text = f.read()

    assert out_text == f'{text}\n{text}\n{text}\n'


@pytest.mark.parametrize(
    ('text', 'expected'),
    (
        ('Hello.\n', ['Hello.']),
        ('Hello, there!', ['Hello, there!']),
        ('Hello.\nSentence two.', ['Hello.', 'Sentence two.']),
        ('1. 2. 3. 4.', ['1.', '2.', '3.', '4.']),
    ),
)
def test_get_sent_tokenizer(text: str, expected: list[str]) -> None:
    tokenizer = get_sent_tokenizer()

    assert tokenizer(text) == expected


def test_read_documents(tmp_path: pathlib.Path) -> None:
    expected = [
        b'Lorem ipsum dolor sit amet.\nConsectetur adipiscing elit.',
        b'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
        b'Ut enim ad minim veniam, quis nostrud.',
    ]

    file0 = tmp_path / 'file0.txt'
    with open(file0, 'wb') as f:
        f.write(b'\n\n\n')

    assert read_documents_bytes(file0) == []

    file1 = tmp_path / 'file1.txt'
    with open(file1, 'wb') as f:
        f.write(expected[0])
        f.write(b'\n\n')

    (found_item_0,) = read_documents_bytes([file0, file1])
    assert found_item_0 == expected[0]

    file2 = tmp_path / 'file2.txt'
    with open(file2, 'wb') as f:
        f.write(expected[1])
        f.write(b'\n\n')
        f.write(expected[2])

    found = read_documents_bytes([file0, file1, file2])
    assert found == expected


def test_write_documents(tmp_path: pathlib.Path) -> None:
    out = tmp_path / 'out.txt'
    docs = [
        'Lorem ipsum dolor sit amet. Consectetur adipiscing elit.',
        'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
        'Ut enim ad minim veniam, quis nostrud.',
    ]
    expected = """\
Lorem ipsum dolor sit amet.
Consectetur adipiscing elit.

Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Ut enim ad minim veniam, quis nostrud.

"""

    write_documents(out, docs)
    with open(out) as f:
        text = f.read()
    assert text == expected
