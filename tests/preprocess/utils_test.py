from __future__ import annotations

import pathlib
import tarfile

import pytest

from llm.preprocess.utils import safe_extract


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
