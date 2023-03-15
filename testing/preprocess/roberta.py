from __future__ import annotations

import pathlib
import random
import string

import tokenizers
from tokenizers.implementations.base_tokenizer import BaseTokenizer


def get_alphabet_tokenizer(max_seq_len: int) -> BaseTokenizer:
    vocab = ['[PAD]', '[UNK]', '[SEP]', '[CLS]', '[MASK]']
    vocab.extend(list(string.ascii_letters))
    tokenizer = tokenizers.BertWordPieceTokenizer(
        vocab={char: index for index, char in enumerate(vocab)},
        lowercase=False,
    )
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_length=max_seq_len)
    return tokenizer


def generate_document(sequences: int, max_seq_len: int) -> list[list[str]]:
    document: list[list[str]] = []
    for _ in range(sequences):
        k = random.randint(min(4, max_seq_len), max_seq_len)
        document.append(random.choices(list(string.ascii_letters), k=k))
    return document


def write_document_shards(
    output_dir: pathlib.Path | str,
    num_files: int = 8,
    num_documents: int = 10,
    document_length: int = 100,
    document_seq_length: int = 64,
) -> list[pathlib.Path]:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        generate_document(document_length, document_seq_length)
        for _ in range(num_documents)
    ]
    for file_index in range(num_files):
        with open(output_dir / f'{file_index}.txt', 'w') as f:
            for document_index in range(file_index, len(documents), num_files):
                for line in documents[document_index]:
                    f.write(' '.join(line))
                f.write('\n\n')

    return list(output_dir.glob('*.txt'))
