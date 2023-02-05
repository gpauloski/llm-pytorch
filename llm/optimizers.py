from __future__ import annotations

import logging
from typing import Any
from typing import Iterable
from typing import Literal

import torch

try:  # pragma: no cover
    from apex.optimizer import FusedAdam
    from apex.optimizers import FusedLAMB

    APEX_IMPORT_ERROR = None
except ImportError as e:
    APEX_IMPORT_ERROR = e

logger = logging.getLogger(__name__)


def get_optimizer(
    name: Literal['lamb', 'adam'],
    params: Iterable[torch.Tensor | dict[str, Any]],
    lr: float,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    if name == 'adam':  # pragma: no cover
        if APEX_IMPORT_ERROR is None:
            optimizer = FusedAdam(params, lr=lr, **kwargs)
        else:
            logger.warning(
                'NVIDIA Apex is not installed so defaulting to native PyTorch '
                'Adam. Better performance can be enabled with Apex\'s '
                'FusedAdam.',
            )
            optimizer = torch.optim.Adam(params, lr=lr, **kwargs)
    elif name == 'lamb':  # pragma: no cover
        if APEX_IMPORT_ERROR is None:
            optimizer = FusedLAMB(params, lr=lr, **kwargs)
        else:
            raise ImportError(
                'FusedLamb is not available. NVIDIA Apex is not installed.',
            ) from APEX_IMPORT_ERROR
    else:
        raise ValueError(f'Unknown optimizer: {name}')

    return optimizer
