"""RoBERTa pretraining encoder.

This implements the DOC-SENTENCES sampling strategy of RoBERTa. Samples are
not pre-masked as in Devlin et al. and are rather dynamically masked at
runtime as in RoBERTa. Next sentence prediction is also not used.

```bash
python -m llm.preprocess.roberta --help
```
"""

from __future__ import annotations

import functools
import logging
import multiprocessing
import os
import pathlib
import random
import time
import warnings
from collections.abc import Sequence

import click
import h5py
import tokenizers
from tokenizers.implementations.base_tokenizer import BaseTokenizer

from llm.utils import init_logging

logger = logging.getLogger('roberta-encoder')

SequenceT = list[str]
SentenceT = list[str]
DocumentT = list[SentenceT]


def read_documents(
    input_file: pathlib.Path | str,
    tokenizer: BaseTokenizer,
) -> list[DocumentT]:
    documents: list[DocumentT] = []
    document: DocumentT = []

    max_length = (
        None
        if tokenizer.truncation is None
        else tokenizer.truncation['max_length']
    )
    tokenizer.no_padding()
    tokenizer.no_truncation()

    with open(input_file) as f:
        for line_ in f.readlines():
            line = line_.strip()
            if len(line) == 0 and len(document) > 0:
                documents.append(document)
                document = []
            elif len(line) > 0:
                document.append(
                    tokenizer.encode(line, add_special_tokens=False).tokens,
                )

    if len(document) > 0:  # pragma: no branch
        documents.append(document)

    tokenizer.enable_padding()
    tokenizer.enable_truncation(max_length=max_length)

    return documents


def create_samples_from_document(
    document: DocumentT,
    max_seq_len: int,
    short_seq_prob: float,
) -> list[SequenceT]:
    samples: list[SequenceT] = []

    if random.random() < short_seq_prob:
        max_tokens = random.randint(2, max_seq_len - 2)
    else:
        max_tokens = max_seq_len - 2

    for sentence in document:
        if len(samples) == 0 or len(samples[-1]) > max_tokens:
            # Starting new sample and randomly update max seq length
            samples.append([])
            if random.random() < short_seq_prob:
                max_tokens = random.randint(2, max_seq_len - 2)
            else:
                max_tokens = max_seq_len - 2

        samples[-1].extend(sentence)

    return samples


def create_samples_from_documents(
    documents: list[DocumentT],
    max_seq_len: int,
    short_seq_prob: float,
) -> list[SequenceT]:
    samples: list[SequenceT] = []

    for _i, document in enumerate(documents):
        document_samples = create_samples_from_document(
            document,
            max_seq_len=max_seq_len,
            short_seq_prob=short_seq_prob,
        )
        samples.extend(document_samples)

    return samples


def write_samples(
    samples: list[tokenizers.Encoding],
    output_file: str | pathlib.Path,
) -> None:
    input_ids = [sample.ids for sample in samples]
    attention_masks = [sample.attention_mask for sample in samples]
    special_tokens_masks = [sample.special_tokens_mask for sample in samples]

    with h5py.File(output_file, 'w') as f:
        f.create_dataset(
            'input_ids',
            data=input_ids,
            dtype='i4',
            compression='gzip',
        )
        f.create_dataset(
            'attention_masks',
            data=attention_masks,
            dtype='i1',
            compression='gzip',
        )
        f.create_dataset(
            'special_tokens_masks',
            data=special_tokens_masks,
            dtype='i1',
            compression='gzip',
        )


def encode_file(
    input_file: str | pathlib.Path,
    output_file: str | pathlib.Path,
    tokenizer: str | BaseTokenizer,
    *,
    max_seq_len: int = 512,
    short_seq_prob: float = 0.0,
) -> None:
    start = time.monotonic()
    logger.info(f'Encoding {input_file}...')

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if isinstance(tokenizer, str):  # pragma: no cover
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer)
    tokenizer.enable_padding(length=max_seq_len)
    tokenizer.enable_truncation(max_length=max_seq_len)

    documents = read_documents(input_file, tokenizer)
    samples = create_samples_from_documents(
        documents,
        max_seq_len=max_seq_len,
        short_seq_prob=short_seq_prob,
    )
    random.shuffle(samples)

    encoded_samples = tokenizer.encode_batch(samples, is_pretokenized=True)
    write_samples(encoded_samples, output_file)

    total_time = time.monotonic() - start
    logger.info(f'Finished {output_file} in {total_time:.0f} seconds')


def encode_files(
    input_files: Sequence[str | pathlib.Path],
    output_dir: str | pathlib.Path,
    tokenizer: str,
    *,
    max_seq_len: int = 512,
    short_seq_prob: float = 0.0,
    processes: int = 4,
) -> None:
    start = time.monotonic()

    logger.info(f'Preparing to encode {len(input_files)} shards...')
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Writing output shards to {output_dir}')

    zfill = len(str(len(input_files)))

    _encode = functools.partial(
        encode_file,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        short_seq_prob=short_seq_prob,
    )

    input_files = sorted(input_files)
    args = [
        (input_file, output_dir / f'shard-{str(i).zfill(zfill)}.hdf5')
        for i, input_file in enumerate(input_files)
    ]

    logger.info(f'Starting multiprocessing pool with {processes} workers...')
    if processes > 1:  # pragma: no branch
        warnings.warn(
            'Using tokenizers with processes > 1 can lead to deadlocks. '
            'If you encounter a deadlock, try again with a single process.',
            stacklevel=2,
        )
    with multiprocessing.Pool(processes=processes, maxtasksperchild=1) as pool:
        pool.starmap(_encode, args)

        # Graceful shutdown for coverage
        pool.close()
        pool.join()

    total_time = time.monotonic() - start
    logger.info(f'Finished processing in {total_time:.0f} seconds')


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
    '-t',
    '--tokenizer',
    metavar='PATH',
    required=True,
    help='Path to trained tokenizer to load.',
)
@click.option(
    '-l',
    '--max-seq-len',
    type=int,
    default=512,
    help='Maximum sequence length.',
)
@click.option(
    '-s',
    '--short-seq-prob',
    type=float,
    default=0.1,
    help='Probablity to create shorter sequences.',
)
@click.option(
    '-p',
    '--processes',
    type=int,
    default=4,
    help='Number of processes for concurrent shard encoding.',
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
    tokenizer: str,
    max_seq_len: int,
    short_seq_prob: float,
    processes: int,
    log_level: str,
    rich: bool,
) -> None:
    """Encode FILEPATHS for RoBERTa pretraining."""
    init_logging(log_level, rich=rich)

    # This silences the error:
    #
    # The current process just got forked, after parallelism has already been
    # used. Disabling parallelism to avoid deadlocks...
    # To disable this warning, you can either:
    #   - Avoid using `tokenizers` before the fork if possible
    #   - Explicitly set the environment variable TOKENIZERS_PARALLELISM=false
    #
    # Note we set this in a few places to ensure the environment variable
    # is set in subprocesses.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    encode_files(
        input_files=inputs,
        output_dir=output_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        short_seq_prob=short_seq_prob,
        processes=processes,
    )


if __name__ == '__main__':
    cli()
