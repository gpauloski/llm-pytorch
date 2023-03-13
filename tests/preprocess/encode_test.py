from __future__ import annotations

import pathlib
import random
import string

import h5py
import pytest
import tokenizers

from llm.preprocess.encode import CLS_TOKEN
from llm.preprocess.encode import PAD_TOKEN
from llm.preprocess.encode import SEP_TOKEN
from llm.preprocess.encode import Sample
from llm.preprocess.encode import create_samples_from_document
from llm.preprocess.encode import create_samples_from_documents
from llm.preprocess.encode import encode_files
from llm.preprocess.encode import read_documents
from llm.preprocess.encode import write_samples

VOCAB = list(string.ascii_letters) + ['[UNK]', '[SEP]', '[CLS]', '[PAD]']
ALPHABET_TOKENIZER = tokenizers.BertWordPieceTokenizer(
    vocab={char: index for index, char in enumerate(VOCAB)},
    lowercase=False,
)


def generate_document(sequences: int, max_seq_len: int) -> list[list[str]]:
    document: list[list[str]] = []
    for _ in range(sequences):
        k = random.randint(min(4, max_seq_len), max_seq_len)
        document.append(random.choices(list(string.ascii_letters), k=k))
    return document


@pytest.mark.parametrize(
    ('segment', 'expected_tokens', 'special_token_positions', 'max_seq_len'),
    (
        ([], [CLS_TOKEN, SEP_TOKEN], [0, 1], 2),
        (['A'], [CLS_TOKEN, 'A', SEP_TOKEN, PAD_TOKEN, PAD_TOKEN], [0, 2], 5),
        (['A', 'B', 'C'], [CLS_TOKEN, 'A', 'B', SEP_TOKEN], [0, 3], 4),
    ),
)
def test_single_sample_segment(
    segment: list[str],
    expected_tokens: list[str],
    special_token_positions: list[int],
    max_seq_len: int,
) -> None:
    sample = Sample.from_single_segment(segment, max_seq_len)
    assert sample.tokens == expected_tokens
    assert sample.special_token_positions == special_token_positions
    assert not sample.next_sentence_label


def test_single_sample_segment_bad_max_seq_len() -> None:
    with pytest.raises(ValueError, match='^Max sequence length'):
        Sample.from_single_segment([], max_seq_len=1)


@pytest.mark.parametrize(
    (
        'first_segment',
        'second_segment',
        'expected_tokens',
        'special_token_positions',
        'max_seq_len',
    ),
    (
        ([], [], [CLS_TOKEN, SEP_TOKEN, SEP_TOKEN], [0, 1, 2], 3),
        (
            ['A'],
            ['B', 'C'],
            [CLS_TOKEN, 'A', SEP_TOKEN, 'B', 'C', SEP_TOKEN, PAD_TOKEN],
            [0, 2, 5],
            7,
        ),
        (
            ['A'],
            ['B', 'C', 'D'],
            [CLS_TOKEN, 'A', SEP_TOKEN, 'B', 'C', SEP_TOKEN],
            [0, 2, 5],
            6,
        ),
    ),
)
def test_two_sample_segment(
    first_segment: list[str],
    second_segment: list[str],
    expected_tokens: list[str],
    special_token_positions: list[int],
    max_seq_len: int,
) -> None:
    sample = Sample.from_two_segments(
        first_segment,
        second_segment,
        True,
        max_seq_len,
    )
    assert sample.tokens == expected_tokens
    assert sample.special_token_positions == special_token_positions
    assert sample.next_sentence_label


def test_two_sample_segment_bad_max_seq_len() -> None:
    with pytest.raises(ValueError, match='^Max sequence length'):
        Sample.from_two_segments([], [], True, max_seq_len=2)


def test_two_sample_segment_segment_truncated() -> None:
    with pytest.raises(ValueError, match='the second segment being removed'):
        Sample.from_two_segments(['A'], ['B'], True, max_seq_len=4)


def test_read_documents(tmp_path: pathlib.Path) -> None:
    filepath = tmp_path / 'text.txt'
    with open(filepath, 'w') as f:
        f.write('A B C D E\nA B C\n\nA B C D\n\n\n\nA B C')

    documents = read_documents(filepath, ALPHABET_TOKENIZER)
    expected_documents = [
        [['A', 'B', 'C', 'D', 'E'], ['A', 'B', 'C']],
        [['A', 'B', 'C', 'D']],
        [['A', 'B', 'C']],
    ]
    assert documents == expected_documents


def test_create_samples_from_document() -> None:
    document = generate_document(100, 64)
    tokens = sum(len(line) for line in document)

    samples = create_samples_from_document(document, 128, 0.5)
    sample_tokens = sum(len(sample.tokens) for sample in samples)

    # Add lower bound just to make sure samples aren't ending up emptyish
    assert sample_tokens / 4 < tokens < sample_tokens


def test_create_samples_from_documents() -> None:
    documents = [generate_document(100, 64) for _ in range(10)]
    tokens = sum(sum(len(line) for line in document) for document in documents)

    samples = create_samples_from_documents(documents, 128, 0.5)
    sample_tokens = sum(len(sample.tokens) for sample in samples)

    assert sample_tokens / 2 < tokens < sample_tokens


def test_write_samples(tmp_path: pathlib.Path) -> None:
    document = generate_document(100, 64)
    samples = create_samples_from_document(document, 128, 0.5)

    filepath = tmp_path / 'samples.hdf5'
    write_samples(samples, filepath, ALPHABET_TOKENIZER)

    with h5py.File(filepath, 'r') as f:
        input_ids = f['input_ids'][:]
        special_token_positions = f['special_token_positions'][:]
        next_sentence_labels = f['next_sentence_labels'][:]

    assert (
        len(input_ids)
        == len(special_token_positions)
        == len(next_sentence_labels)
        == len(samples)
    )


def test_encode_files(tmp_path: pathlib.Path) -> None:
    input_dir = tmp_path / 'input'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / 'output_dir'

    documents = [generate_document(100, 64) for _ in range(10)]
    files = 8
    for file_index in range(files):
        with open(input_dir / f'{file_index}.txt', 'w') as f:
            for document_index in range(file_index, len(documents), files):
                for line in documents[document_index]:
                    f.write(' '.join(line))
                f.write('\n\n')

    input_files = list(input_dir.glob('*.txt'))

    encode_files(
        input_files,
        output_dir,
        ALPHABET_TOKENIZER,
        max_seq_len=128,
        short_seq_prob=0.2,
        processes=2,
    )

    assert len(list(output_dir.glob('*.hdf5')))
