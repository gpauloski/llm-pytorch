from __future__ import annotations

import logging
import os
import pathlib
from typing import Callable
from typing import Iterable

logger = logging.getLogger(__name__)
NLTK_DOWNLOAD_DIR = str(pathlib.Path.home() / '.cache/nltk_data')
os.environ['NLTK_DATA'] = NLTK_DOWNLOAD_DIR

# Must import after setting the environment variable
import nltk  # noqa: E402


def combine_document_files(
    filepaths: Iterable[pathlib.Path | str],
    output_file: pathlib.Path | str,
) -> None:
    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(exist_ok=True)

    sent_tokenizer = get_sent_tokenizer()

    with open(output_file, 'w') as target:
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                document_lines = f.readlines()
                sentences = []
                for line in document_lines:
                    sentences.extend(sent_tokenizer(line.strip()))
                target.write('\n'.join(sentences))
                target.write('\n\n')


def get_sent_tokenizer() -> Callable[[str], list[str]]:
    downloader = nltk.downloader.Downloader()
    if not downloader.is_installed('punkt'):  # pragma: no cover
        nltk.download('punkt', quiet=True)
        download_dir = downloader.default_download_dir()
        logger.info(f'Downloaded NLTK punkt model to {download_dir}.')

    return nltk.tokenize.sent_tokenize


def write_documents(path: pathlib.Path | str, documents: list[str]) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True)

    sent_tokenizer = get_sent_tokenizer()

    with open(path, 'w') as f:
        for document in documents:
            document = document.replace('\n', ' ')
            sentences = sent_tokenizer(document)
            f.write('\n'.join(sentences))
            f.write('\n\n')
