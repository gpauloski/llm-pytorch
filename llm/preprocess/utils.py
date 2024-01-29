"""Preprocessing script utilities."""

from __future__ import annotations

import decimal
import pathlib
import re
import tarfile


def readable_to_bytes(size: str) -> int:
    """Convert string with bytes units to the integer value of bytes.

    Source: [ProxyStore](https://github.com/proxystore/proxystore/blob/79dfdc0fc2c5a5093cf5e57b2ecf61c48fde6773/proxystore/utils.py#L130){target=_blank}

    Example:
        ```python
        >>> readable_to_bytes('1.2 KB')
        1200
        >>> readable_to_bytes('0.6 MiB')
        629146
        ```

    Args:
        size: String to parse for bytes size.

    Returns:
        Integer number of bytes parsed from the string.

    Raises:
        ValueError: If the input string contains more than two parts
            (i.e., a value and a unit).
        ValueError: If the unit is not one of KB, MB, GB, TB, KiB, MiB, GiB,
            or TiB.
        ValueError: If the value cannot be cast to a float.
    """
    units_to_bytes = dict(
        b=1,
        kb=int(1e3),
        mb=int(1e6),
        gb=int(1e9),
        tb=int(1e12),
        kib=int(2**10),
        mib=int(2**20),
        gib=int(2**30),
        tib=int(2**40),
    )

    # Try casting size to value (will only work if no units)
    try:
        return int(float(size))
    except ValueError:
        pass

    # Ensure space between value and unit
    size = re.sub(r'([a-zA-Z]+)', r' \1', size.strip())

    parts = [s.strip() for s in size.split()]
    if len(parts) != 2:
        raise ValueError(
            'Input string "{size}" must contain only a value and a unit.',
        )

    value, unit = parts

    try:
        value_size = decimal.Decimal(value)
    except decimal.InvalidOperation as e:
        raise ValueError(f'Unable to interpret "{value}" as a float.') from e
    try:
        unit_size = units_to_bytes[unit.lower()]
    except KeyError as e:
        raise ValueError(f'Unknown unit type {unit}.') from e

    return int(value_size * unit_size)


def safe_extract(name: pathlib.Path | str, target: pathlib.Path | str) -> None:
    """Safely extract a tar file.

    Note:
        This extraction method is designed to safeguard against CVE-2007-4559.

    Args:
        name: Path to tar file to extract.
        target: Target path to extract to.
    """
    name = pathlib.Path(name).resolve().absolute()
    target = pathlib.Path(target).resolve().absolute()

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
