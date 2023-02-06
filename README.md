# LLM Training Scripts

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gpauloski/llm-pytorch/main.svg)](https://results.pre-commit.ci/latest/github/gpauloski/llm-pytorch/main)
[![Tests](https://github.com/gpauloski/llm-pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/gpauloski/llm-pytorch/actions)

Tools and training scripts for large language models built using PyTorch.
This repository is the successor to my old training tools [BERT-PyTorch](https://github.com/gpauloski/BERT-PyTorch) as the old code had a lot of technical debt and was not well tested.
This repository provides (or will provide):
- data preprocessing scripts
- training scripts
- training guides

with better code health and maintainability thanks to tests, type checking, linters, documentation, etc.

## Table of Contents

- [Install](#install)
- [Training](#training)

## Install

This package is Linux only and requires Python >=3.9.
It is recommended to install the package in a virtual environment.
```bash
$ python -m venv venv     # or $ virtualenv venv
$ . venv/bin/activate
$ pip install torch       # torch install instructions may differ
$ pip install .           # use -e for editable mode
```
PyTorch installation instructions vary by system and CUDA versions so check the latest instructions [here](https://pytorch.org/get-started/locally/).
NVIDIA Apex can be installed to use the `FusedAdam` and `FusedLAMB` optimizers.
See the directions [here](https://github.com/NVIDIA/apex#from-source).

### System Specific Guides

Installation guides for specific clusters:

- [Polaris Install](guides/polaris-install.md)

### Development Install

If you plan to develop within the `llm` package and contribute changes,
we use Tox for testing and pre-commit for linting.
Tox can be used to configure a development environment.
```bash
$ tox --devenv venv -e py310
$ . venv/bin/activate
$ pre-commit install
```
There are also `{py39,py310}-gpu` environments that install `torch` with CUDA.
The `{py39,py310}` CPU PyTorch environments are the ones used in CI.

Alternatively, a development environment can be built from scratch (if you do not have Tox installed already).
```bash
$ python -m venv venv
$ . venv/bin/activate
$ pip install torch
$ pip install -e .[dev]
$ pre-commit install
```

## Training

The available training scripts are executable modules in `llm.trainers`.
E.g., get started with `python -m llm.trainers.bert --help`.
Each training script is slightly different so guides are provided in [`guides/`](guides/):

- [BERT Pretraining](guides/bert-pretraining.md)
