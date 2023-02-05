from __future__ import annotations

import logging

from llm.trainers.bert.main import main

logger = logging.getLogger('llm.trainers.bert')


if __name__ == '__main__':
    try:
        ret = main()
    except Exception as e:
        logger.exception(e)

    raise SystemExit(ret)
