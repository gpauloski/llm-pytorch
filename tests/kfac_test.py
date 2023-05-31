from __future__ import annotations

import argparse

from llm.kfac import add_kfac_options


def test_add_kfac_options_to_parser() -> None:
    parser = argparse.ArgumentParser()
    add_kfac_options(parser)
    args = parser.parse_args(
        [
            '--kfac',
            '--kfac-factor-update-steps',
            '1',
            '--kfac-inv-update-steps',
            '2',
        ],
    )
    assert args.kfac
    assert args.kfac_factor_update_steps == 1
    assert args.kfac_inv_update_steps == 2
