from __future__ import annotations

import time

import torch


class Timer:
    def __init__(self) -> None:
        self._history: list[float] = []
        self._start_time = time.time()

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.time()

    def stop(self) -> float:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - self._start_time
        self._history.append(elapsed)
        return elapsed

    def lap(self) -> float:
        lap_time = self.stop()
        self.start()
        return lap_time

    def get_history(self) -> list[float]:
        return self._history

    def get_history_sum(self) -> float:
        return sum(self._history)

    def get_history_mean(self) -> float:
        return sum(self._history) / len(self._history)
