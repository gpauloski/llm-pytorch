"""Pretraining vocabulary builder.

```bash
python -m llm.preprocess.vocab --help
```
"""
from __future__ import annotations

import glob
import logging
import pathlib
from typing import Literal
from typing import Sequence

import click

from llm.preprocess.tokenize import get_tokenizer
from llm.utils import init_logging

logger = logging.getLogger('sharder')


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
        f'(lowercase: {lowercase}).',
    )

    if special_tokens is not None and len(special_tokens) > 0:
        tokenizer.add_special_tokens(special_tokens)
        logger.info(f'Added special tokens to tokenizer: {special_tokens}.')

    # trainer does not work with PosixPaths
    input_files = [str(f) for f in input_files]

    logger.info(f'Training tokenizer on {len(input_files)} files...')
    tokenizer.train(input_files, vocab_size=size, show_progress=True)
    logger.info('Training completed.')

    output_file = pathlib.Path(output_file).resolve()
    output_file.parent.mkdir(exist_ok=True, parents=True)

    # Get vocab sorted by frequency
    vocab_with_freq = tokenizer.get_vocab(with_added_tokens=True)
    vocab_sorted = [(word, freq) for word, freq in vocab_with_freq.items()]
    vocab_sorted.sort(key=lambda x: x[1])
    vocab = [word for word, _ in vocab_sorted]

    # Move special tokens to front
    if special_tokens is not None and len(special_tokens) > 0:
        for token in reversed(special_tokens):
            vocab.insert(0, vocab.pop(vocab.index(token)))

    with open(output_file, 'w') as f:
        f.write('\n'.join(vocab))
    logger.info(f'Vocab file written to {output_file}.')


@click.command()
@click.option(
    '--input',
    metavar='PATH',
    required=True,
    help='Glob of input text files to build vocab from.',
)
@click.option(
    '--output',
    metavar='PATH',
    required=True,
    help='Output filepath for vocabulary.',
)
@click.option(
    '--size',
    default=30522,
    type=int,
    help='Size of vocabulary.',
)
@click.option(
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
    multiple=True,
    default=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
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
    input: str,  # noqa: A002
    output: str,
    size: int,
    tokenizer: Literal['bpe', 'wordpiece'],
    cased: bool,
    special_token: list[str],
    log_level: str,
    rich: bool,
) -> None:
    """Pre-training vocabulary builder.

    Arguments default to the standard uncased BERT with wordpiece method.
    """
    init_logging(log_level, rich=rich)

    input_files = glob.glob(input)

    train_vocab(
        tokenizer,
        input_files=input_files,
        output_file=output,
        size=size,
        lowercase=not cased,
        special_tokens=special_token,
    )


if __name__ == '__main__':
    cli()
