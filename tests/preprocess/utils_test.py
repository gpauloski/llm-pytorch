from __future__ import annotations

import pathlib
import tarfile

import pytest

from llm.preprocess.utils import readable_to_bytes
from llm.preprocess.utils import safe_extract


@pytest.mark.parametrize(
    ('value', 'expected'),
    (
        ('0 B', 0),
        ('1 B', 1),
        ('1 KB', int(1e3)),
        ('1 MB', int(1e6)),
        ('1 GB', int(1e9)),
        ('1 TB', int(1e12)),
        ('1KB', int(1e3)),
        ('1MB', int(1e6)),
        ('1GB', int(1e9)),
        ('1TB', int(1e12)),
        ('1000 TB', int(1e15)),
        ('1.001 KB', 1001),
        ('1.0001 KB', 1000),
        ('123.457 MB', 123457000),
        ('40 TB', 40000000000000),
        ('0', 0),
        ('1', 1),
        ('1000000000', 1000000000),
    ),
)
def test_readable_to_bytes(value: str, expected: int) -> None:
    assert readable_to_bytes(value) == expected


def test_readable_to_bytes_too_many_parts() -> None:
    with pytest.raises(ValueError, match='value and a unit'):
        readable_to_bytes('1 B GB')

    with pytest.raises(ValueError, match='value and a unit'):
        readable_to_bytes('GB')


def test_readable_to_bytes_unknown_unit() -> None:
    with pytest.raises(ValueError, match='Unknown unit'):
        readable_to_bytes('1 XB')


def test_readable_to_bytes_value_cast_failure() -> None:
    with pytest.raises(ValueError, match='float'):
        # Note that is letter o rather than zero
        readable_to_bytes('O B')


def test_safe_extract(tmp_path: pathlib.Path) -> None:
    src_dir = tmp_path / 'src'
    src_dir.mkdir()

    tar_path = tmp_path / 'out.tar.gz'

    with tarfile.open(tar_path, 'w:gz') as f:
        for i in range(3):
            txt_file = src_dir / f'{i}.txt'
            with open(txt_file, 'w') as t:
                t.write('abc')
            f.add(txt_file, arcname=txt_file.name)

    target_dir = tmp_path / 'target'
    safe_extract(tar_path, target_dir)

    assert len(list(target_dir.glob('*.txt'))) == 3


def test_safe_extract_path_traversal(tmp_path: pathlib.Path) -> None:
    src_dir = tmp_path / 'src'
    src_dir.mkdir()

    tar_path = tmp_path / 'out.tar.gz'

    txt_file = src_dir / 'test.txt'
    with open(txt_file, 'w') as f:
        f.write('abc')

    with tarfile.open(tar_path, 'w:gz') as f:
        # arcname has parent ".." prefixed
        f.add(txt_file, arcname='../test.txt')

    target_dir = tmp_path / 'target'
    with pytest.raises(OSError, match='^Tarfile contains'):
        safe_extract(tar_path, target_dir)
