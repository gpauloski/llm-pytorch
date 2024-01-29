"""BERT-Large pretraining config using large-batches with LAMB.

Hyperparameters are based on:
https://github.com/NVIDIA/DeepLearningExamples/blob/ca5ae20e3d1af3464159754f758768052c41c607/PyTorch/LanguageModeling/BERT/scripts/configs/pretrain_config.sh
"""

from __future__ import annotations

import os

import torch

from llm.models import bert as bert_models
from llm.trainers.bert.data import NvidiaBertDatasetConfig

PHASE = 1
BERT_CONFIG = bert_models.BERT_LARGE
OPTIMIZER = 'lamb'
GRADIENT_CHECKPOINTING = False
OUTPUT_DIR = 'runs/bert-large-pretraining'
RUN_NAME = f'phase-{PHASE}'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f'checkpoints/{RUN_NAME}')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, f'tensorboard/{RUN_NAME}')
LOG_FILE = os.path.join(OUTPUT_DIR, f'logs/{RUN_NAME}.txt')
DTYPE = torch.float16
SEED = 42

# ACCUMULATION_STEPS is computed automatically by llm.trainers.bert
if PHASE == 1:
    DATASET_CONFIG = NvidiaBertDatasetConfig(
        '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/',
    )

    GLOBAL_BATCH_SIZE = 65536
    BATCH_SIZE = 64
    STEPS = 7038
    CHECKPOINT_STEPS = 500

    LR = 6e-3
    WARMUP_STEPS = 2000
elif PHASE == 2:
    DATASET_CONFIG = NvidiaBertDatasetConfig(
        '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/',
    )

    GLOBAL_BATCH_SIZE = 32768
    BATCH_SIZE = 16
    STEPS = 1563
    CHECKPOINT_STEPS = 200

    LR = 4e-3
    WARMUP_STEPS = 200
else:
    raise NotImplementedError
