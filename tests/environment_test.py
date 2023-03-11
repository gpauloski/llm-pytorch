from __future__ import annotations

import logging

from llm.environment import log_environment


def test_log_environment(caplog):
    with caplog.at_level(logging.DEBUG):
        log_environment(logging.DEBUG)

    assert 'Runtime environment:' in caplog.text
