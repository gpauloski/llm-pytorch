"""Pretraining text formatting utilities."""
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
    """Combine multiple text files into one."""
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
    """Get a sentence tokenizer.

    Returns:
        An NLTK sentence tokenizer.
    """
    downloader = nltk.downloader.Downloader()
    if not downloader.is_installed('punkt'):  # pragma: no cover
        nltk.download('punkt', quiet=True)
        download_dir = downloader.default_download_dir()
        logger.info(f'Downloaded NLTK punkt model to {download_dir}.')

    return nltk.tokenize.sent_tokenize


def read_documents_bytes(
    files: Iterable[pathlib.Path | str] | pathlib.Path | str,
) -> list[bytes]:
    """Read documents from files.

    Args:
        files: List of files containing documents separated by blank lines
            to read.

    Returns:
        List of documents where each document is the read bytestring.
    """
    if not isinstance(files, Iterable):
        files = [files]

    documents: list[bytes] = []
    document_lines: list[bytes] = []

    for current_file in files:
        with open(current_file, 'rb') as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0 and len(document_lines) > 0:
                    documents.append(b'\n'.join(document_lines))
                    document_lines = []
                elif len(line) > 0:
                    document_lines.append(line)

    if len(document_lines) > 0:
        documents.append(b'\n'.join(document_lines))

    return documents


def write_documents(path: pathlib.Path | str, documents: list[str]) -> None:
    """Write a list of documents to a file.

    Args:
        path: Path to write documents to.
        documents: Documents to write. Each document will be separated by
            a blank line.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True)

    sent_tokenizer = get_sent_tokenizer()

    with open(path, 'w') as f:
        for document in documents:
            document = document.replace('\n', ' ')
            sentences = sent_tokenizer(document)
            f.write('\n'.join(sentences))
            f.write('\n\n')
