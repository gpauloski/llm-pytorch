"""Performance timer."""

from __future__ import annotations

import time

import torch


class Timer:
    """Performance timer.

    Args:
        synchronize: Synchronize CUDA workers before and after the timer.
    """

    def __init__(self, synchronize: bool = False) -> None:
        self._synchronize = synchronize
        self._history: list[float] = []
        self._start_time = time.time()

    def start(self) -> None:
        """Start the timer."""
        if torch.cuda.is_available() and self._synchronize:
            torch.cuda.synchronize()
        self._start_time = time.time()

    def stop(self) -> float:
        """Stop the timer and return the elapsed time."""
        if torch.cuda.is_available() and self._synchronize:
            torch.cuda.synchronize()

        elapsed = time.time() - self._start_time
        self._history.append(elapsed)
        return elapsed

    def lap(self) -> float:
        """Log a lap of the timer."""
        lap_time = self.stop()
        self.start()
        return lap_time

    def get_history(self) -> list[float]:
        """Get the history of all lap times."""
        return self._history

    def get_history_sum(self) -> float:
        """Get the sum of all lap times."""
        return sum(self._history)

    def get_history_mean(self) -> float:
        """Get the mean of all lap times."""
        return sum(self._history) / len(self._history)
