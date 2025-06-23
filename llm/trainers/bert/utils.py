"""BERT pretraining utilities."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any
from typing import get_type_hints
from typing import Literal
from typing import Optional
from typing import Union

import torch
import transformers

from llm.checkpoint import load_checkpoint
from llm.checkpoint import save_checkpoint
from llm.config import Config
from llm.trainers.bert.data import NvidiaBertDatasetConfig
from llm.trainers.bert.data import RobertaDatasetConfig
from llm.utils import gradient_accumulation_steps

logger = logging.getLogger('llm.trainers.bert')


@dataclasses.dataclass
class TrainingConfig:
    """Training configuration."""

    PHASE: int
    BERT_CONFIG: dict[str, Any]
    OPTIMIZER: Literal['adam', 'lamb']
    CHECKPOINT_DIR: str
    TENSORBOARD_DIR: str
    DATASET_CONFIG: Union[  # noqa: UP007
        NvidiaBertDatasetConfig,
        RobertaDatasetConfig,
    ]
    GLOBAL_BATCH_SIZE: int
    BATCH_SIZE: int
    STEPS: int
    CHECKPOINT_STEPS: int
    LR: float
    WARMUP_STEPS: int

    # Computed options
    ACCUMULATION_STEPS: int

    # Optional options
    CLIP_GRAD_NORM: Optional[float] = None
    DTYPE: Optional[torch.dtype] = None
    GRADIENT_CHECKPOINTING: bool = False
    LOG_FILE: Optional[str] = None
    SEED: int = 42


def parse_config(config: Config) -> TrainingConfig:
    """Parses a config ensuring all required options are present."""
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
                f'Unable to verify config option {field_name}: {field_type}\n'
                f'{e}',
            )

        if not match:
            raise TypeError(
                f'Expected config entry {field_name} to be type '
                f'{field_type} but got {type(config[field_name])}.',
            )

    # Only take args that are fields of TrainingConfig
    fields = {field.name for field in dataclasses.fields(TrainingConfig)}
    config_ = {k: v for k, v in config.items() if k in fields}

    return TrainingConfig(**config_)


def checkpoint(
    config: TrainingConfig,
    global_step: int,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    sampler_index: int = 0,
) -> None:
    """Write a training checkpoint."""
    if torch.distributed.get_rank() == 0:
        # Extract from possible AMPModel
        model = model._model if hasattr(model, '_model') else model
        # Extract from possible DistributedDataParallel
        model = model.module if hasattr(model, 'module') else model

        save_checkpoint(
            checkpoint_dir=config.CHECKPOINT_DIR,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            phase=config.PHASE,
            sampler_index=sampler_index,
        )
    logger.info(
        f'Saved checkpoint at global step {global_step}',
        extra={'ranks': [0]},
    )


def load_state(
    config: TrainingConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[int, int, int]:
    """Load the latest checkpoint if one exists.

    Returns:
        Tuple of the global step, epoch, and sampler index to resume
        (or start) from.
    """
    global_step = 0
    epoch = 1
    sampler_index = 0

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    checkpoint = load_checkpoint(
        config.CHECKPOINT_DIR,
        map_location='cpu',  # next(model.parameters()).device,
    )
    if checkpoint is None:
        logger.info(
            'No checkpoint found to resume from',
            extra={'ranks': [0]},
        )
        return global_step, epoch, sampler_index

    logger.info(
        f'Loaded checkpoint from {checkpoint.filepath}',
        extra={'ranks': [0]},
    )
    # Load model to the model and not the AMP wrapper
    model = model._model if hasattr(model, '_model') else model
    # Load model to the model and not the DDP wrapper
    model = model.module if hasattr(model, 'module') else model
    model.load_state_dict(checkpoint.model_state_dict)

    if checkpoint.kwargs['phase'] == config.PHASE:
        logger.info(
            'Checkpoint from current phase. Loading training state',
            extra={'ranks': [0]},
        )

        if (  # pragma: no branch
            checkpoint.optimizer_state_dict is not None
            and optimizer is not None
        ):
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)

        if (  # pragma: no branch
            checkpoint.scheduler_state_dict is not None
            and scheduler is not None
        ):
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)

        global_step = checkpoint.global_step
        epoch = checkpoint.kwargs['epoch']
        sampler_index = (
            checkpoint.kwargs['sampler_index']
            if 'sampler_index' in checkpoint.kwargs
            else 0
        )
    else:
        logger.info(
            'Checkpoint from new phase. Resetting training state',
            extra={'ranks': [0]},
        )

    return global_step, epoch, sampler_index


def get_optimizer_grouped_parameters(
    model: transformers.BertForPreTraining,
) -> list[dict[str, Any]]:
    """Get the parameters of the BERT model to optimizer."""
    # configure the weight decay for bert models
    params = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']

    params_decay = [p for n, p in params if not any(v in n for v in no_decay)]
    params_no_decay = [p for n, p in params if any(v in n for v in no_decay)]

    return [
        {'params': params_decay, 'weight_decay': 0.01},
        {'params': params_no_decay, 'weight_decay': 0.0},
    ]
