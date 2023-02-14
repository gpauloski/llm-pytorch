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
