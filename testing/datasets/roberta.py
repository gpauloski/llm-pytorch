from __future__ import annotations

import pathlib
import tempfile

from tokenizers.implementations.base_tokenizer import BaseTokenizer

from llm.preprocess.roberta import encode_file
from testing.preprocess.roberta import get_alphabet_tokenizer
from testing.preprocess.roberta import write_document_shards


def write_roberta_shard(
    filepath: pathlib.Path | str,
    max_seq_len: int,
) -> BaseTokenizer:
    tokenizer = get_alphabet_tokenizer(max_seq_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        (input_file,) = write_document_shards(tmpdir, num_files=1)

        encode_file(
            input_file,
            filepath,
            tokenizer,
            max_seq_len=max_seq_len,
            short_seq_prob=0.2,
        )

    return tokenizer
