# LLM Training Scripts

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gpauloski/llm-pytorch/main.svg)](https://results.pre-commit.ci/latest/github/gpauloski/llm-pytorch/main)
[![Tests](https://github.com/gpauloski/llm-pytorch/actions/workflows/tests.yml/badge.svg)](https://github.com/gpauloski/llm-pytorch/actions)


## Install

This package is Linux only and requires Python >=3.9.
It is recommended to install the package in a virtual environment.
```
$ python -m venv venv     # or $ virtualenv venv
$ . venv/bin/activate
$ pip install torch       # torch install instructions may differ
$ pip install .           # use -e for editable mode
```
PyTorch installation instructions vary and the latests commands can be found [here](https://pytorch.org/get-started/locally/).
NVIDIA Apex can be install to use `FusedAdam` and `FusedLAMB` optimizers.
See the directions [here](https://github.com/NVIDIA/apex#from-source).

### Development Install

If you plan to develop within the `llm` package and contribute changes,
we use Tox for testing and pre-commit for linting.
Tox can be used to configure a development environment.
```
$ tox --devenv venv -e py310
$ . venv/bin/activate
$ pre-commit install
```
There are also `{py39,py310}-gpu` environments that install `torch` with CUDA.

## Guides

- [BERT Pretraining](guides/bert-pretraining.md)
- [Polaris Install](guides/polaris-install.md)
