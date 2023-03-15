from __future__ import annotations

from typing import Literal

import pytest
import tokenizers

from llm.preprocess.tokenize import get_tokenizer


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
