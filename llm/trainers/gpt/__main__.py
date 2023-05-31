from __future__ import annotations

import logging

from llm.trainers.gpt.main import main

logger = logging.getLogger('llm.trainers.gpt')


if __name__ == '__main__':
    try:
        ret = main()
    except Exception as e:
        logger.exception(e)
        ret = 1

    raise SystemExit(ret)
