"""BERT-Base pretraining config.

Hyperparameters are based on the original BERT paper
https://arxiv.org/pdf/1810.04805.pdf but the global batch size and iterations
are scaled to enable distributed training following Table 3 of the RoBERTa
paper https://arxiv.org/pdf/1907.11692.pdf.

Batch size is 8192 and training is done for 31250 steps (equivalent to the
original 256 batch size for 1M steps). The first 90% of steps are done
with sequence length 128 and the last 10% are done with sequence length 512.
"""
from __future__ import annotations

import os

import torch

from llm.models import bert as bert_models

PHASE = 1
BERT_CONFIG = bert_models.BERT_BASE
OPTIMIZER = 'adam'
GRADIENT_CHECKPOINTING = False
OUTPUT_DIR = 'runs/bert-base-pretraining'
RUN_NAME = f'phase-{PHASE}'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f'checkpoints/{RUN_NAME}')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, f'tensorboard/{RUN_NAME}')
LOG_FILE = os.path.join(OUTPUT_DIR, f'logs/{RUN_NAME}.txt')
CLIP_GRAD_NORM = 1.0
DTYPE = torch.float16
SEED = 42

# ACCUMULATION_STEPS is computed automatically by llm.trainers.bert
if PHASE == 1:
    MAX_SEQ_LENGTH = 128
    DATA_DIR = '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_128_max_pred_20_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/'  # noqa: E501

    GLOBAL_BATCH_SIZE = 8192
    BATCH_SIZE = 256
    STEPS = 28125
    CHECKPOINT_STEPS = 500

    LR = 1e-4
    WARMUP_STEPS = int(0.01 * STEPS)
elif PHASE == 2:
    MAX_SEQ_LENGTH = 512
    DATA_DIR = '/grand/SuperBERT/jgpaul/datasets/encoded/wikibooks/nvidia_static_masked_30K/hdf5_lower_case_1_seq_len_512_max_pred_80_masked_lm_prob_0.15_random_seed_12345_dupe_factor_5/books_wiki_en_corpus/'  # noqa: E501

    GLOBAL_BATCH_SIZE = 8192
    BATCH_SIZE = 32
    STEPS = 3125
    CHECKPOINT_STEPS = 200

    LR = 1e-4
    WARMUP_STEPS = int(0.01 * STEPS)
else:
    raise NotImplementedError
