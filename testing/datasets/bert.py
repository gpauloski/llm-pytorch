from __future__ import annotations

import pathlib
import random
from typing import NamedTuple

import h5py


class NvidiaBertShard(NamedTuple):
    input_ids: list[list[int]]
    input_mask: list[list[bool]]
    segment_ids: list[list[bool]]
    masked_lm_positions: list[list[int]]
    masked_lm_ids: list[list[int]]
    next_sentence_labels: list[bool]


def generate_int_sample(seq_len: int, imin: int, imax: int) -> list[int]:
    return [random.randint(imin, imax - 1) for _ in range(seq_len)]


def generate_bool_sample(seq_len: int) -> list[bool]:
    return [random.choice((True, False)) for _ in range(seq_len)]


def generate_nvidia_bert_dataset(
    samples: int,
    seq_len: int,
    vocab_size: int = 30000,
) -> NvidiaBertShard:
    return NvidiaBertShard(
        [generate_int_sample(seq_len, 0, vocab_size) for _ in range(samples)],
        [generate_bool_sample(seq_len) for _ in range(samples)],
        [generate_bool_sample(seq_len) for _ in range(samples)],
        [generate_int_sample(seq_len, 0, seq_len) for _ in range(samples)],
        [generate_int_sample(seq_len, 0, vocab_size) for _ in range(samples)],
        generate_bool_sample(samples),
    )


def write_nvidia_bert_shard(
    filepath: pathlib.Path | str,
    samples: int,
    seq_len: int,
    vocab_size: int = 30000,
) -> None:
    shard = generate_nvidia_bert_dataset(samples, seq_len, vocab_size)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('input_ids', data=shard.input_ids, dtype='i4')
        f.create_dataset('input_mask', data=shard.input_mask, dtype='i1')
        f.create_dataset('segment_ids', data=shard.segment_ids, dtype='i1')
        f.create_dataset(
            'masked_lm_positions',
            data=shard.masked_lm_positions,
            dtype='i4',
        )
        f.create_dataset('masked_lm_ids', data=shard.masked_lm_ids, dtype='i4')
        f.create_dataset(
            'next_sentence_labels',
            data=shard.next_sentence_labels,
            dtype='i1',
        )
