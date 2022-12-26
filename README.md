# LLM Training Scripts using Colossal-AI and PyTorch

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gpauloski/llm-colossal-ai/main.svg)](https://results.pre-commit.ci/latest/github/gpauloski/llm-colossal-ai/main)
[![Tests](https://github.com/gpauloski/llm-colossal-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/gpauloski/llm-colossal-ai/actions)


## Install

This package is Linux only and requires CUDA >=11.3.
It is recommended to install the package in a virtual environment.
```
$ python -m venv venv     # or $ virtualenv venv
$ . venv/bin/activate
$ pip install -r requirements.txt
$ pip install .           # use -e for editable mode
```

### Development Install

If you plan to develop within the `llm` package and contribute changes,
we use Tox for testing and pre-commit for linting.
Tox can be used to configure a development environment.
```
$ tox --devenv venv -e py310
$ . venv/bin/activate
$ pre-commit install
```

## Guides

- [BERT Pretraining](guides/bert-pretraining.md)
