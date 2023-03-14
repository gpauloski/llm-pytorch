from __future__ import annotations

import pathlib
import random
import string

import h5py
import tokenizers

from llm.preprocess.roberta import create_samples_from_document
from llm.preprocess.roberta import create_samples_from_documents
from llm.preprocess.roberta import encode_files
from llm.preprocess.roberta import read_documents
from llm.preprocess.roberta import write_samples

VOCAB = ['[PAD]', '[UNK]', '[SEP]', '[CLS]'] + list(string.ascii_letters)
ALPHABET_TOKENIZER = tokenizers.BertWordPieceTokenizer(
    vocab={char: index for index, char in enumerate(VOCAB)},
    lowercase=False,
)
MAX_SEQ_LEN = 128
ALPHABET_TOKENIZER.enable_padding(length=MAX_SEQ_LEN)
ALPHABET_TOKENIZER.enable_truncation(max_length=MAX_SEQ_LEN)


def generate_document(sequences: int, max_seq_len: int) -> list[list[str]]:
    document: list[list[str]] = []
    for _ in range(sequences):
        k = random.randint(min(4, max_seq_len), max_seq_len)
        document.append(random.choices(list(string.ascii_letters), k=k))
    return document


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
    document = generate_document(100, 64)
    samples = create_samples_from_document(document, MAX_SEQ_LEN, 0.5)
    encoded_samples = ALPHABET_TOKENIZER.encode_batch(
        samples,
        is_pretokenized=True,
    )

    filepath = tmp_path / 'samples.hdf5'
    write_samples(encoded_samples, filepath)

    with h5py.File(filepath, 'r') as f:
        input_ids = f['input_ids'][:]
        attention_masks = f['attention_masks'][:]

    assert len(input_ids) == len(attention_masks) == len(samples)
    assert input_ids.shape == (len(samples), MAX_SEQ_LEN)
    assert attention_masks.shape == (len(samples), MAX_SEQ_LEN)


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
        max_seq_len=MAX_SEQ_LEN,
        short_seq_prob=0.2,
        processes=2,
    )

    assert len(list(output_dir.glob('*.hdf5')))
