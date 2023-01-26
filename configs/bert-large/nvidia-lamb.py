"""BERT-Large pretraining config using large-batches with LAMB.

Hyperparameters are based on:
https://github.com/NVIDIA/DeepLearningExamples/blob/ca5ae20e3d1af3464159754f758768052c41c607/PyTorch/LanguageModeling/BERT/scripts/configs/pretrain_config.sh
"""
from __future__ import annotations

import os

from colossalai.amp import AMP_TYPE

from llm.models import bert as bert_models

PHASE = 1
BERT_CONFIG = bert_models.BERT_LARGE
OPTIMIZER = 'lamb'
GRADIENT_CHECKPOINTING = False
OUTPUT_DIR = 'runs/bert-large-pretraining'
RUN_NAME = f'phase-{PHASE}'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f'checkpoints/{RUN_NAME}')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, f'tensorboard/{RUN_NAME}')

if PHASE == 1:
    MAX_SEQ_LENGTH = 128
    DATA_DIR = '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/'  # noqa: E501

    GLOBAL_BATCH_SIZE = 65536
    BATCH_SIZE = 64
    STEPS = 7038
    CHECKPOINT_STEPS = 500

    LR = 6e-3
    WARMUP_STEPS = 2000
elif PHASE == 2:
    MAX_SEQ_LENGTH = 512
    DATA_DIR = '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/'  # noqa: E501

    GLOBAL_BATCH_SIZE = 32768
    BATCH_SIZE = 16
    STEPS = 1563
    CHECKPOINT_STEPS = 200

    LR = 4e-3
    WARMUP_STEPS = 200
else:
    raise NotImplementedError

# Colossal-AI options
# accumulation_steps is computed automatically by llm.trainers.bert
seed = 42
clip_grad_norm = 1.0
fp16 = dict(mode=AMP_TYPE.TORCH)