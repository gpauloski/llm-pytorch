from __future__ import annotations

from typing import Any
from typing import Literal

import tokenizers


def get_tokenizer(
    kind: Literal['wordpiece', 'bpe'],
    vocab: dict[str, int] | str | None = None,
    lowercase: bool = False,
    padding_options: dict[str, Any] | None = None,
    truncation_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> tokenizers.implementations.base_tokenizer.BaseTokenizer:
    if kind == 'wordpiece':
        tokenizer = tokenizers.BertWordPieceTokenizer(
            vocab=vocab,
            clean_text=True,
            handle_chinese_chars=True,
            lowercase=lowercase,
            **kwargs,
        )
    elif kind == 'bpe':
        tokenizer = tokenizers.ByteLevelBPETokenizer(
            vocab=vocab,
            lowercase=lowercase,
            add_prefix_space=True,
            trim_offsets=True,
            **kwargs,
        )
    else:
        raise AssertionError(f'Unsupported kind "{kind}."')

    if padding_options is not None:
        tokenizer.enable_padding(**padding_options)
    if truncation_options is not None:
        tokenizer.enable_truncation(**truncation_options)

    return tokenizer
