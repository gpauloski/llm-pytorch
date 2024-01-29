"""Pretraining tokenizer trainer.

```bash
python -m llm.preprocess.tokenizer --help
```
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence
from typing import Any
from typing import Literal

import click
import tokenizers

from llm.utils import init_logging

logger = logging.getLogger('tokenizer')


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


def train_vocab(
    kind: Literal['bpe', 'wordpiece'],
    input_files: Sequence[pathlib.Path | str],
    output_file: pathlib.Path | str,
    size: int,
    lowercase: bool = False,
    special_tokens: list[str] | None = None,
) -> None:
    tokenizer = get_tokenizer(kind, lowercase=lowercase)
    logger.info(
        f'Initialized {tokenizer.__class__.__name__} tokenizer '
        f'(lowercase: {lowercase})',
    )

    # trainer does not work with PosixPaths
    input_files = [str(f) for f in input_files]

    logger.info(f'Training tokenizer on {len(input_files)} files...')
    logger.info(f'Using special tokens: {special_tokens}')
    tokenizer.train(
        input_files,
        vocab_size=size,
        show_progress=True,
        special_tokens=special_tokens,
    )
    logger.info('Training completed')

    tokenizer.save(str(output_file), pretty=True)
    logger.info(f'Tokenizer saved to {output_file}')


@click.command()
@click.argument(
    'inputs',
    metavar='FILEPATHS',
    required=True,
    nargs=-1,
)
@click.option(
    '-o',
    '--output-file',
    metavar='PATH',
    required=True,
    help='Output file to save serialized tokenizer to.',
)
@click.option(
    '-s',
    '--size',
    default=30522,
    type=int,
    help='Size of vocabulary.',
)
@click.option(
    '-t',
    '--tokenizer',
    type=click.Choice(['bpe', 'wordpiece'], case_sensitive=False),
    default='wordpiece',
    help='Tokenizer type.',
)
@click.option(
    '--cased/--uncased',
    default=False,
    help='Vocab/tokenizer is case-sensitive.',
)
@click.option(
    '-s',
    '--special-token',
    default=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    metavar='TOKEN',
    multiple=True,
    help='Special tokens to prepend to vocab.',
)
@click.option(
    '--log-level',
    default='INFO',
    type=click.Choice(
        ['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        case_sensitive=False,
    ),
    help='Minimum logging level.',
)
@click.option(
    '--rich/--no-rich',
    default=False,
    help='Use rich output formatting.',
)
def cli(
    inputs: tuple[str],
    output_file: str,
    size: int,
    tokenizer: Literal['bpe', 'wordpiece'],
    cased: bool,
    special_token: list[str],
    log_level: str,
    rich: bool,
) -> None:
    """Train a tokenizer on FILEPATHS.

    Arguments default to the standard uncased BERT with wordpiece method.
    """
    init_logging(log_level, rich=rich)

    train_vocab(
        tokenizer,
        input_files=inputs,
        output_file=output_file,
        size=size,
        lowercase=not cased,
        special_tokens=list(special_token),
    )


if __name__ == '__main__':
    cli()
