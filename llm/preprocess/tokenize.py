"""Tokenizer utilities."""
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
    """Get a tokenizer by name.

    Args:
        kind: Tokenizer name to create.
        vocab: Vocabulary file or dictionary to initialized tokenizer with.
        lowercase: Set the tokenizer to lowercase.
        padding_options: Enable padding options on the tokenizer.
        truncation_options: Enable truncation options on the tokenizer.
        kwargs: Additional arguments to pass to the tokenizer.

    Returns:
        A tokenizer instance.
    """
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
