from __future__ import annotations

import argparse
import logging
import pathlib
import sys
from typing import Literal
from typing import Sequence

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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='llm.preprocess.vocab',
        description='Vocabulary builder',
        usage='python -m llm.preprocess.vocab --help',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input',
        nargs='+',
        help='input text files to generate vocabulary from',
    )
    parser.add_argument(
        '--output',
        help='output vocabulary file',
    )
    parser.add_argument(
        '--size',
        type=int,
        default=30522,
        help='size of vocabulary',
    )
    parser.add_argument(
        '--tokenizer',
        choices=['bpe', 'wordpiece'],
        default='wordpiece',
        help='tokenizer type',
    )
    parser.add_argument(
        '--cased',
        default=False,
        action='store_true',
        help='build a case-sensitive vocabulary',
    )
    parser.add_argument(
        '--special-tokens',
        nargs='*',
        default=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
        help='special tokens to put at front of vocab',
    )
    parser.add_argument(
        '--loglevel',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='minimum logging level',
    )
    parser.add_argument(
        '--no-rich',
        action='store_true',
        help='disable rich stdout formatting',
    )
    args = parser.parse_args(argv)

    init_logging(args.loglevel, rich=not args.no_rich)

    train_vocab(
        args.tokenizer,
        input_files=args.input,
        output_file=args.output,
        size=args.size,
        lowercase=not args.cased,
        special_tokens=args.special_tokens,
    )

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
