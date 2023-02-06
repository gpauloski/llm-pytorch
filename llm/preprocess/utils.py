from __future__ import annotations

import pathlib
import tarfile


def safe_extract(name: pathlib.Path | str, target: pathlib.Path | str) -> None:
    name = pathlib.Path(name).absolute()
    target = pathlib.Path(target).absolute()

    with tarfile.open(name, 'r:gz') as f:
        # CVE-2007-4559 input sanitation
        for member in f.getmembers():
            member_path = (target / member.name).resolve()
            if target not in member_path.parents:
                raise OSError(
                    'Tarfile contains member that extracts outside of the '
                    'target directory. This could be a path traversal attack.',
                )
        f.extractall(target)
