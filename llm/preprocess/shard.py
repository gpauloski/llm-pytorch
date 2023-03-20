"""Pretraining test sharder.

```bash
python -m llm.preprocess.shard --help
```
"""
from __future__ import annotations

import logging
import pathlib
import random
import sys
from typing import Generator
from typing import Iterable
from typing import Sequence

import click

from llm.preprocess.format import read_documents_bytes
from llm.preprocess.utils import readable_to_bytes
from llm.utils import init_logging

logger = logging.getLogger('sharder')


def get_document_shards(
    documents: Iterable[bytes],
    max_bytes_per_shard: int,
) -> Generator[list[bytes], None, None]:
    current_documents: list[bytes] = []
    current_size = 0

    for document in documents:
        size = sys.getsizeof(document)
        if (
            len(current_documents) == 0
            or current_size + size <= max_bytes_per_shard
        ):
            current_documents.append(document)
            current_size += size
        else:
            yield current_documents
            current_documents = [document]
            current_size = size

    assert len(current_documents) != 0
    yield current_documents


def shard(
    input_files: Sequence[pathlib.Path | str],
    output_dir: pathlib.Path | str,
    shard_format: str,
    max_bytes_per_shard: int,
    shuffle: bool = False,
) -> None:
    if '{index}' not in shard_format:
        raise ValueError(
            'The pattern \'{index}\' is not in the provided shard format '
            f'\'{shard_format}\'.',
        )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info('Reading documents from input files...')
    documents = read_documents_bytes(input_files)
    logger.info(f'Loaded {len(documents)} documents.')

    if shuffle:
        random.shuffle(documents)
        logger.info('Shuffled documents.')

    logger.info('Creating shards from documents...')
    shards = list(get_document_shards(documents, max_bytes_per_shard))
    zfill = len(str(len(shards)))

    logger.info('Writing shards...')
    for i, shard in enumerate(shards):
        filepath = output_dir / shard_format.format(index=str(i).zfill(zfill))
        with open(filepath, 'wb') as f:
            for document in shard:
                f.write(document)
                f.write(b'\n\n')
        logger.info(f'Wrote shard {i} to {filepath}.')

    logger.info('Completed.')


@click.command()
@click.argument(
    'inputs',
    metavar='FILEPATHS',
    required=True,
    nargs=-1,
)
@click.option(
    '-o',
    '--output-dir',
    metavar='PATH',
    required=True,
    help='Output directory for encoded shards.',
)
@click.option(
    '-s',
    '--size',
    metavar='SIZE',
    required=True,
    help='Max data size of each shard.',
)
@click.option(
    '-f',
    '--format',
    default='shard-{index}.txt',
    help='Shard name format where {index} is replaced by shard index.',
)
@click.option(
    '--shuffle/--no-shuffle',
    default=False,
    help='Shuffle documents before sharding.',
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
    output_dir: str,
    size: str,
    format: str,  # noqa: A002
    shuffle: bool,
    log_level: str,
    rich: bool,
) -> None:
    """Shard documents in FILEPATHS into equally sized files."""
    init_logging(log_level, rich=rich)

    size_bytes = readable_to_bytes(size)

    shard(inputs, output_dir, format, size_bytes, shuffle)


if __name__ == '__main__':
    cli()
