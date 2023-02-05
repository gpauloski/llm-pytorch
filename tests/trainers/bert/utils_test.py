from __future__ import annotations

import pathlib
from typing import Generator
from unittest import mock

import pytest
import torch

from llm.config import Config
from llm.models.bert import from_config
from llm.trainers.bert.utils import checkpoint
from llm.trainers.bert.utils import get_optimizer_grouped_parameters
from llm.trainers.bert.utils import load_state
from llm.trainers.bert.utils import parse_config

TINY_CONFIG = dict(
    vocab_size=1000,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=64,
    hidden_act='gelu',
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=64,
    type_vocab_size=2,
    initializer_range=0.02,
)


@pytest.fixture
def config(tmp_path: pathlib.Path) -> Generator[Config, None, None]:
    config_ = dict(
        PHASE=0,
        BERT_CONFIG={},
        OPTIMIZER='adam',
        CHECKPOINT_DIR=str(tmp_path / 'checkpoint'),
        TENSORBOARD_DIR=str(tmp_path / 'tensorboard'),
        DATA_DIR=str(tmp_path / 'data'),
        GLOBAL_BATCH_SIZE=128,
        BATCH_SIZE=16,
        STEPS=1000,
        CHECKPOINT_STEPS=100,
        LR=0.1,
        WARMUP_STEPS=50,
    )
    with mock.patch('torch.distributed.get_world_size', return_value=1):
        yield Config(**config_)


def test_parse_config(config: Config) -> None:
    training_config = parse_config(config)

    assert training_config.ACCUMULATION_STEPS == (
        config.GLOBAL_BATCH_SIZE // config.BATCH_SIZE
    )


def test_parse_config_missing_entry(config: Config) -> None:
    del config['PHASE']

    with pytest.raises(TypeError, match='PHASE'):
        parse_config(config)


def test_parse_config_bad_type(config: Config) -> None:
    config.PHASE = '0'

    with pytest.raises(TypeError, match='str'):
        parse_config(config)


def test_checkpoint(config: Config):
    training_config = parse_config(config)

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    path = pathlib.Path(training_config.CHECKPOINT_DIR)
    path.mkdir()

    with mock.patch('torch.distributed.get_rank') as mock_get_rank:
        mock_get_rank.return_value = 1
        checkpoint(
            training_config,
            global_step=0,
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert len(list(path.iterdir())) == 0

        mock_get_rank.return_value = 0
        checkpoint(
            training_config,
            global_step=0,
            epoch=0,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        assert len(list(path.iterdir())) == 1


def test_load_state_none(config: Config):
    train_config = parse_config(config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    global_step, epoch = load_state(train_config, model, optimizer, scheduler)
    assert global_step == 0
    assert epoch == 0


def test_load_state_same_phase(config: Config):
    train_config = parse_config(config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    with mock.patch('torch.distributed.get_rank', return_value=0):
        checkpoint(
            train_config,
            global_step=1,
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    global_step, epoch = load_state(train_config, model, optimizer, scheduler)
    assert global_step == 1
    assert epoch == 2


def test_load_state_different_phase(config: Config):
    train_config = parse_config(config)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    with mock.patch('torch.distributed.get_rank', return_value=0):
        checkpoint(
            train_config,
            global_step=1,
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    train_config.PHASE += 1
    global_step, epoch = load_state(train_config, model, optimizer, scheduler)
    assert global_step == 0
    assert epoch == 0


def test_get_optimizer_grouped_parameters() -> None:
    model = from_config(config=TINY_CONFIG)
    get_optimizer_grouped_parameters(model)
