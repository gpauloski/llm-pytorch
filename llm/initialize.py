from __future__ import annotations

import argparse
import logging
import os
import pathlib
import random

import numpy
import torch

from llm.config import Config
from llm.config import load_config
from llm.utils import init_logging

logger = logging.getLogger(__name__)


def get_default_parser(
    prog: str | None = None,
    description: str | None = None,
    usage: str | None = None,
) -> argparse.ArgumentParser:
    """Get default argument parser to be used with initialize_from_args().

    Args:
        prog: optional name of program.
        description: optional description of program.
        usage: optional program usage.

    Returns:
        ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        usage=usage,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', help='path to config file')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='single worker distributed configuration for debugging',
    )
    parser.add_argument(
        '--loglevel',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='minimum logging level',
    )
    parser.add_argument(
        '--rich',
        action='store_true',
        help='use rich for pretty stdout logging',
    )
    return parser


def initialize(
    *,
    debug: bool = False,
    loglevel: int | str = 'INFO',
    logfile: pathlib.Path | str | None = None,
    seed: int | None = None,
    rich: bool = False,
) -> None:
    """Initialize the distributed context.

    Perform the following: 1) initialize logging, 2) initialized torch
    distributed, and 3) set the cuda device is available.

    Args:
        debug: initialize torch distributed for debugging (single worker).
        loglevel: minimum logging level.
        logfile: log filepath.
        seed: random seed.
        rich: use rich formatting for stdout logging.
    """
    init_logging(loglevel, logfile=logfile, rich=rich, distributed=True)

    backend = 'nccl' if torch.cuda.is_available() else 'gloo'

    if debug:
        os.environ['LOCAL_RANK'] = '0'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'
        torch.distributed.init_process_group(
            backend=backend,
            world_size=1,
            rank=0,
        )
    else:
        torch.distributed.init_process_group(backend=backend)

    logger.info(
        'Distributed initialization complete: '
        f'backend={torch.distributed.get_backend()}, '
        f'world_size={torch.distributed.get_world_size()}',
        extra={'ranks': [0]},
    )

    if torch.cuda.is_available():
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)

        logger.info(
            f'Initialized CUDA local device to {local_rank}',
            extra={'ranks': [0]},
        )

    if seed is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        random.seed(seed + local_rank)
        numpy.random.seed(seed + local_rank)
        torch.manual_seed(seed + local_rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + local_rank)


def initialize_from_args(args: argparse.Namespace) -> Config:
    """Load config and initialize from args."""
    config = load_config(args.config)

    initialize(
        debug=args.debug,
        loglevel=args.loglevel or config.get('LOG_LEVEL', None),
        logfile=config.get('LOG_FILE', None),
        seed=config.get('SEED', None),
        rich=args.rich,
    )

    return config
