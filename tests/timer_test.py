from __future__ import annotations

import time
from unittest import mock

from llm.timer import Timer


def test_timer() -> None:
    timer = Timer()
    laps = 5
    sleep = 0.001

    timer.start()
    for _ in range(laps):
        time.sleep(sleep)
        lap = timer.lap()
        assert lap >= sleep

    time.sleep(sleep)
    timer.stop()

    assert len(timer.get_history()) == laps + 1
    assert timer.get_history_sum() > sleep * (laps + 1)
    assert timer.get_history_mean() > sleep


def test_timer_cuda() -> None:
    with mock.patch('torch.cuda.is_available', return_value=True):
        with mock.patch('torch.cuda.synchronize'):
            timer = Timer()
            timer.start()
            timer.stop()
