from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest import mock

from colossalai.context.config import Config
from colossalai.core import global_context as gpc


@contextmanager
def mock_global_context(**kwargs: Any) -> Generator[None, None, None]:
    with mock.patch.object(gpc, '_config', Config(kwargs)):
        yield
