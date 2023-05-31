from __future__ import annotations

import argparse
import re
import sys


def add_kfac_options(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments for K-FAC.

    Args:
        parser: Parser object to add K-FAC arguments to.
    """
    args_str = ' '.join(sys.argv)
    group = parser.add_argument_group(
        title='K-FAC',
        description='K-FAC preconditioning options',
    )
    group.add_argument(
        '--kfac',
        action='store_true',
        help='Use K-FAC for preconditioning',
    )
    group.add_argument(
        '--kfac-factor-update-steps',
        type=int,
        required=bool(re.search(r'--kfac($|\s)', args_str)),
        help='Steps between updating Kronecker factors',
    )
    group.add_argument(
        '--kfac-inv-update-steps',
        type=int,
        required=bool(re.search(r'--kfac($|\s)', args_str)),
        help='Steps between recomputing inverses/eigen decompositions',
    )
    group.add_argument(
        '--kfac-damping',
        type=float,
        default=0.001,
        help='K-FAC damping value',
    )
    group.add_argument(
        '--kfac-factor-decay',
        type=float,
        default=0.95,
        help='K-FAC factor decay rate',
    )
    group.add_argument(
        '--kfac-kl-clip',
        type=float,
        default=0.001,
        help='K-FAC KL-clip',
    )
