from __future__ import annotations

import pathlib

import h5py

from llm.preprocess.roberta import create_samples_from_document
from llm.preprocess.roberta import create_samples_from_documents
from llm.preprocess.roberta import encode_files
from llm.preprocess.roberta import read_documents
from llm.preprocess.roberta import write_samples
from testing.preprocess.roberta import generate_document
from testing.preprocess.roberta import get_alphabet_tokenizer
from testing.preprocess.roberta import write_document_shards

MAX_SEQ_LEN = 128


def test_read_documents(tmp_path: pathlib.Path) -> None:
    tokenizer = get_alphabet_tokenizer(MAX_SEQ_LEN)

    filepath = tmp_path / 'text.txt'
    with open(filepath, 'w') as f:
        f.write('A B C D E\nA B C\n\nA B C D\n\n\n\nA B C')

    documents = read_documents(filepath, tokenizer)
    expected_documents = [
        [['A', 'B', 'C', 'D', 'E'], ['A', 'B', 'C']],
        [['A', 'B', 'C', 'D']],
        [['A', 'B', 'C']],
    ]
    assert documents == expected_documents


def test_create_samples_from_document() -> None:
    document = generate_document(100, 64)
    tokens = sum(len(line) for line in document)
    sentences = len(document)

    samples = create_samples_from_document(document, 128, 0.5)
    sample_tokens = sum(len(sample) for sample in samples)

    assert tokens == sample_tokens
    assert sentences > len(samples)


def test_create_samples_from_documents() -> None:
    documents = [generate_document(100, 64) for _ in range(10)]
    tokens = sum(sum(len(line) for line in document) for document in documents)
    sentences = sum(len(document) for document in documents)

    samples = create_samples_from_documents(documents, MAX_SEQ_LEN, 0.5)
    sample_tokens = sum(len(sample) for sample in samples)

    assert tokens == sample_tokens
    assert sentences > len(samples)


def test_write_samples(tmp_path: pathlib.Path) -> None:
    tokenizer = get_alphabet_tokenizer(MAX_SEQ_LEN)

    document = generate_document(100, 64)
    samples = create_samples_from_document(document, MAX_SEQ_LEN, 0.5)
    encoded_samples = tokenizer.encode_batch(samples, is_pretokenized=True)

    filepath = tmp_path / 'samples.hdf5'
    write_samples(encoded_samples, filepath)

    with h5py.File(filepath, 'r') as f:
        input_ids = f['input_ids'][:]
        attention_masks = f['attention_masks'][:]
        special_tokens_masks = f['special_tokens_masks'][:]

    assert len(input_ids) == len(attention_masks) == len(samples)
    assert input_ids.shape == (len(samples), MAX_SEQ_LEN)
    assert attention_masks.shape == (len(samples), MAX_SEQ_LEN)
    assert special_tokens_masks.shape == (len(samples), MAX_SEQ_LEN)


def test_encode_files(tmp_path: pathlib.Path) -> None:
    tokenizer = get_alphabet_tokenizer(MAX_SEQ_LEN)

    input_dir = tmp_path / 'input'
    output_dir = tmp_path / 'output_dir'
    input_files = write_document_shards(input_dir, num_files=8)

    encode_files(
        input_files,
        output_dir,
        tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        short_seq_prob=0.2,
        processes=2,
    )

    assert len(list(output_dir.glob('*.hdf5'))) == 8
