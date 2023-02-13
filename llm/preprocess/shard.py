from __future__ import annotations

import argparse
import logging
import pathlib
import random
import sys
from typing import Generator
from typing import Iterable
from typing import Sequence

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

    logger.info('Writing shards...')
    for i, shard in enumerate(
        get_document_shards(documents, max_bytes_per_shard),
    ):
        output_file = output_dir / shard_format.format(index=i)
        with open(output_file, 'wb') as f:
            for document in shard:
                f.write(document)
                f.write(b'\n\n')
        logger.info(f'Wrote shard {i} to {output_file}.')

    logger.info('Completed.')


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog='llm.preprocess.shard',
        description='Text file sharder',
        usage='python -m llm.preprocess.shard --help',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input',
        nargs='+',
        help='input text files to shard',
    )
    parser.add_argument(
        '--output',
        help='output directory for shards',
    )
    parser.add_argument(
        '--size',
        help='max size (bytes) of each shard',
    )
    parser.add_argument(
        '--format',
        default='shard-{index}.txt',
        help='shard name format where {index} is with the shard index',
    )
    parser.add_argument(
        '--shuffle',
        default=False,
        action='store_true',
        help='shuffle documents in input files before sharding',
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

    size_bytes = readable_to_bytes(args.size)

    shard(args.input, args.output, args.format, size_bytes, args.shuffle)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
