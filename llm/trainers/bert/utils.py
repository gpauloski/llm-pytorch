from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any
from typing import Literal
from typing import get_type_hints

import torch
import transformers

from llm.checkpoint import load_checkpoint
from llm.checkpoint import save_checkpoint
from llm.config import Config
from llm.utils import gradient_accumulation_steps

logger = logging.getLogger('llm.trainers.bert')


@dataclasses.dataclass
class TrainingConfig:
    PHASE: int
    BERT_CONFIG: dict[str, Any]
    OPTIMIZER: Literal['adam', 'lamb']
    CHECKPOINT_DIR: str
    TENSORBOARD_DIR: str
    DATA_DIR: str
    GLOBAL_BATCH_SIZE: int
    BATCH_SIZE: int
    STEPS: int
    CHECKPOINT_STEPS: int
    LR: float
    WARMUP_STEPS: int

    # Computed options
    ACCUMULATION_STEPS: int

    # Optional options
    CLIP_GRAD_NORM: float | None = None
    DTYPE: torch.dtype | None = None
    GRADIENT_CHECKPOINTING: bool = False
    LOG_FILE: str | None = None
    SEED: int = 42


def parse_config(config: Config) -> TrainingConfig:
    config.ACCUMULATION_STEPS = gradient_accumulation_steps(
        global_batch_size=config.GLOBAL_BATCH_SIZE,
        local_batch_size=config.BATCH_SIZE,
        world_size=torch.distributed.get_world_size(),
    )

    for field_name, field_type in get_type_hints(TrainingConfig).items():
        if field_name not in config:
            continue

        try:
            match = isinstance(config[field_name], field_type)
        except TypeError as e:
            # Not all types (GenericAlias types like dict[str, Any]) will
            # support isinstance checks so we just log the error and skip
            match = True
            logger.debug(
                f'Unable to verify config option {field_name}: {field_type}.'
                f'{e}',
            )

        if not match:
            raise TypeError(
                f'Expected config entry {field_name} to be type '
                f'{field_type} but got {type(config[field_name])}.',
            )

    return TrainingConfig(**config)


def checkpoint(
    config: TrainingConfig,
    global_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    if torch.distributed.get_rank() == 0:
        save_checkpoint(
            checkpoint_dir=config.CHECKPOINT_DIR,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            phase=config.PHASE,
        )
    logger.info(
        f'Saved checkpoint at global step {global_step}',
        extra={'ranks': [0]},
    )


def load_state(
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, int]:
    global_step = 0
    epoch = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = load_checkpoint(config.CHECKPOINT_DIR, map_location='cpu')
    if checkpoint is not None:
        logger.info(
            f'Loaded checkpoint from {checkpoint.filepath}',
            extra={'ranks': [0]},
        )
        model.load_state_dict(checkpoint.model_state_dict)

        if checkpoint.kwargs['phase'] == config.PHASE:
            logger.info(
                'Checkpoint from current phase. Loading optimizer state',
                extra={'ranks': [0]},
            )
            if (  # pragma: no branch
                checkpoint.optimizer_state_dict is not None
            ):
                optimizer.load_state_dict(checkpoint.optimizer_state_dict)
                device = (
                    f'cuda:{torch.cuda.current_device()}'
                    if torch.cuda.is_available()
                    else 'cpu'
                )
                for state in optimizer.state.values():
                    for k, v in state.items():  # pragma: no cover
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
            if (  # pragma: no branch
                checkpoint.scheduler_state_dict is not None
            ):
                scheduler.load_state_dict(checkpoint.scheduler_state_dict)
            global_step = checkpoint.global_step
            epoch = checkpoint.kwargs['epoch']
        else:
            logger.info(
                'Checkpoint from new phase. Resetting optimizer state',
                extra={'ranks': [0]},
            )
    else:
        logger.info(
            'No checkpoint found to resume from.',
            extra={'ranks': [0]},
        )

    return global_step, epoch


def get_optimizer_grouped_parameters(
    model: transformers.BertForPreTraining,
) -> list[dict[str, Any]]:
    # configure the weight decay for bert models
    params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    params_decay = [p for n, p in params if not any(v in n for v in no_decay)]
    params_no_decay = [p for n, p in params if any(v in n for v in no_decay)]

    return [
        {'params': params_decay, 'weight_decay': 0.01},
        {'params': params_no_decay, 'weight_decay': 0.0},
    ]
