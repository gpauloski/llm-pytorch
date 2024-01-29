"""LLM package.

Large language model training tools.
Preprocessing scripts are provided in [`llm.preprocess`][llm.preprocess], and
training scripts in [`llm.trainers`][llm.trainers].
"""

from __future__ import annotations

import importlib.metadata
import sys

__version__ = importlib.metadata.version('llm')
