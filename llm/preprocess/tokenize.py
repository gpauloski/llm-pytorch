from __future__ import annotations

from typing import Any
from typing import Literal

import tokenizers


def get_tokenizer(
    kind: Literal['wordpiece', 'bpe'],
    vocab: dict[str, int] | str | None = None,
    lowercase: bool = False,
    **kwargs: Any,
) -> tokenizers.implementations.base_tokenizer.BaseTokenizer:
    if kind == 'wordpiece':
        return tokenizers.BertWordPieceTokenizer(
            vocab=vocab,
            clean_text=True,
            handle_chinese_chars=True,
            lowercase=lowercase,
            **kwargs,
        )
    elif kind == 'bpe':
        return tokenizers.ByteLevelBPETokenizer(
            vocab=vocab,
            lowercase=lowercase,
            add_prefix_space=True,
            trim_offsets=True,
            **kwargs,
        )
    else:
        raise AssertionError(f'Unsupported kind "{kind}."')
