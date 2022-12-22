from __future__ import annotations

import sys
from collections.abc import Sequence

import colossalai
import torch.distributed as dist
from colossalai.core import global_context as gpc


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    argv = argv if argv is not None else sys.argv[1:]
    parser = colossalai.get_default_parser()
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)

    if args.debug:
        colossalai.launch(
            config=args.config,
            rank=0,
            world_size=1,
            host='localhost',
            port=29400,
            backend='gloo',
        )
        args.local_rank = 0
    else:
        colossalai.launch_from_torch(config=args.config)

    colossalai.logging.disable_existing_loggers()
    logger = colossalai.logging.get_dist_logger()
    # TODO: https://github.com/hpcaitech/ColossalAI/issues/2109
    # logger.log_to_file(os.path.join(gpc.config.OUTPUT_DIR, 'logs'))

    logger.info(
        f'Launching training from config at {args.config} '
        f'(debug: {args.debug})',
    )

    if gpc.config.WORKERS != dist.get_world_size():
        logger.warning(
            f'Expected number of workers in the config ({gpc.config.WORKERS}) '
            f'does not match the found number of workers '
            f'({dist.get_world_size()}). This can result in incorrect '
            'gradient accumulation',
        )

    logger.info('Training completed')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
