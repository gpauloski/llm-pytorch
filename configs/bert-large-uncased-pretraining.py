"""BERT pre-training config.

Hyperparameters are based on:
https://github.com/NVIDIA/DeepLearningExamples/blob/ca5ae20e3d1af3464159754f758768052c41c607/PyTorch/LanguageModeling/BERT/scripts/configs/pretrain_config.sh
"""
from __future__ import annotations

from colossalai.amp import AMP_TYPE

from llm.models import bert as bert_models

# Train with 16 nodes, each with 4 GPUs
WORKERS = 16 * 4
PHASE = 2
SEED = 42

BERT_CONFIG = bert_models.BERT_LARGE

OPTIMIZER = 'lamb'


if PHASE == 1:
    MAX_SEQ_LENGTH = 128
    MAX_PREDICTIONS_PER_SEQ = 20
    MASKED_TOKEN_FRACTION = 0.15

    DATA_DIR = ''
    OUTPUT_DIR = 'results/bert-large-phase-1'

    GLOBAL_BATCH_SIZE = 65536
    BATCH_SIZE = 128
    STEPS = 7038

    LR = 6e-3
    WARMUP_STEPS = 2000
elif PHASE == 2:
    MAX_SEQ_LENGTH = 512
    MAX_PREDICTIONS_PER_SEQ = 80
    MASKED_TOKEN_FRACTION = 0.15

    DATA_DIR = ''
    OUTPUT_DIR = 'results/bert-large-phase-2'

    GLOBAL_BATCH_SIZE = 32768
    BATCH_SIZE = 16
    STEPS = 1563

    LR = 4e-3
    WARMUP_STEPS = 200
else:
    raise NotImplementedError

if GLOBAL_BATCH_SIZE % (BATCH_SIZE * WORKERS) != 0:
    raise ValueError(
        'Global batch size must be divisible by the product of the local '
        'batch size and workers',
    )
accumulation_steps = GLOBAL_BATCH_SIZE // (BATCH_SIZE * WORKERS)

fp16 = dict(
    mode=AMP_TYPE.TORCH,
    init_scale=2.0**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=True,
)
