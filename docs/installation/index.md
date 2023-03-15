# Installation

This package is Linux only and requires Python >=3.9.
It is recommended to install the package in a virtual environment of your choice.
```bash
$ python -m venv venv     # or $ virtualenv venv
$ . venv/bin/activate
$ pip install torch       # torch install instructions may differ
$ pip install .           # use -e for editable mode
```
PyTorch installation instructions vary by system and CUDA versions so check the latest instructions [here](https://pytorch.org/get-started/locally/){target=_blank}.

ColossalAI can be installed to use the `FusedAdam` and `FusedLAMB` optimizers.
See the directions [here](https://github.com/hpcaitech/ColossalAI/tree/main#installation){target=_blank}.

## Development Installation

Development installation instructions are provided in the
[Contributing Guide](../contributing/index.md){target=_blank}.

## System Specific Installation

Below are installation guides for specific HPC systems.

- [Polaris](polaris.md)
