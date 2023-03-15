from __future__ import annotations

import logging
import pathlib
import sys
from typing import Literal
from unittest import mock

import click
import datasets
import requests

from llm.preprocess.format import combine_document_files
from llm.preprocess.format import write_documents
from llm.preprocess.utils import safe_extract
from llm.utils import init_logging

logger = logging.getLogger('downloader')

_CACHE_DIR = 'download-cache'


def download_wikipedia(
    output_dir: pathlib.Path | str,
) -> None:  # pragma: no cover
    cache_dir = (pathlib.Path(output_dir) / _CACHE_DIR).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    version = '20220301.en'

    # This function will download a preprocessed dataset so we don't need
    # apache_beam but it will still try and import it
    sys.modules['apache_beam'] = mock.MagicMock()
    dataset = datasets.load_dataset(
        'wikipedia',
        version,
        cache_dir=str(cache_dir),
    )
    del sys.modules['apache_beam']
    logger.info(f'Downloaded wikipedia-{version} to {cache_dir}.')

    logger.info('Formatting and writing dataset...')

    output_file = (
        pathlib.Path(output_dir) / f'wikipedia-{version}.txt'
    ).absolute()
    assert isinstance(dataset, dict)
    documents = [document['text'] for document in dataset['train']]
    write_documents(output_file, documents)

    logger.info(f'Dataset written to {output_file}.')


def download_bookscorpus(
    output_dir: pathlib.Path | str,
) -> None:  # pragma: no cover
    output_dir = pathlib.Path(output_dir).absolute()
    cache_dir = (output_dir / _CACHE_DIR).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = 'https://battle.shawwn.com/sdb/books1/books1.tar.gz'
    download_path = cache_dir / 'bookcorpus.tar.gz'
    target_path = cache_dir / 'bookcorpus'
    output_file = output_dir / 'bookcorpus.txt'

    if not download_path.exists():
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            logger.info(f'Downloading dataset to {download_path}...')
            with open(download_path, 'wb') as f:
                f.write(response.raw.read())
        else:
            logger.error(f'Error downloading dataset: {response.reason}.')
            return
    else:
        logger.info('Using cached dataset.')

    if not target_path.exists():
        logger.info('Extracting dataset...')
        safe_extract(download_path, target_path)
    else:
        logger.info('Using cached extracted dataset.')

    logger.info('Formatting and writing dataset...')
    documents = target_path.glob('books1/epubtxt/**/*.txt')
    combine_document_files(documents, output_file)
    logger.info(f'Dataset written to {output_file}.')


@click.command()
@click.option(
    '--dataset',
    type=click.Choice(['wikipedia', 'bookscorpus'], case_sensitive=False),
    required=True,
    help='Dataset to download.',
)
@click.option(
    '--output',
    metavar='DIR',
    required=True,
    help='Output directory.',
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
    dataset: Literal['wikipedia', 'bookcorpus'],
    output: str,
    log_level: str,
    rich: bool,
) -> None:
    """Pretraining text downloader."""
    init_logging(log_level, rich=rich)

    if dataset == 'wikipedia':
        download_wikipedia(output)
    elif dataset == 'bookscorpus':
        download_bookscorpus(output)
    else:
        raise AssertionError('Unreachable.')


if __name__ == '__main__':
    cli()
