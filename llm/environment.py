"""Utilities for collecting information about the environment.

Tip:
    This module is executable so you can easily check what resources
    your scripts will see as available. This is useful if you need to debug
    what software versions are being used or what hardware is visible by
    PyTorch.
    ```bash
    python -m llm.environment
    ```
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
from collections.abc import Iterable
from typing import NamedTuple

import psutil
import torch
from torch.utils import collect_env

logger = logging.getLogger(__name__)


class Environment(NamedTuple):
    """Named tuple representing collected environment information."""

    os: str
    python_version: str
    python_platform: str
    pip_version: str
    pip_packages: str
    torch_version: str
    torch_is_debug: bool
    cpu_info: str
    total_ram_gb: float
    cuda_is_available: bool
    cuda_compiled_version: str | None
    cuda_runtime_version: str
    cuda_module_loading: str
    nvidia_gpu_models: str
    nvidia_driver: str
    cudnn_version: str


def collect_pip_version() -> str:
    """Collect the pip version."""
    output = subprocess.check_output(['pip', '--version']).decode('utf-8')
    return output.split(' ')[1]


def collect_pip_packages() -> list[str]:
    """Collect a list of relevant pip packages."""
    output = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
    packages = output.split('\n')
    names = [
        'torch',
        'numpy',
        'mypy',
        'colossalai',
        'h5py',
        'tensorboard',
        'tokenizers',
        'transformers',
    ]
    packages = [
        p.strip() for p in packages if any(name in p for name in names)
    ]
    return sorted(packages)


def collect_environment() -> Environment:
    """Collects information on the hardware and software environment."""
    run_lambda = collect_env.run

    bit_count = sys.maxsize.bit_length() + 1
    sys_version = sys.version.replace('\n', ' ')

    pip_version = collect_pip_version()
    pip_packages = collect_pip_packages()
    version_str = torch.__version__
    debug_mode_str = torch.version.debug

    pcores = psutil.cpu_count(logical=False)
    lcores = psutil.cpu_count(logical=True)
    cpu_info = f'{platform.processor()} ({pcores} cores / {lcores} logical)'
    total_ram = round(psutil.virtual_memory().available / 1e9, 2)

    cuda_available_str = torch.cuda.is_available()
    cuda_version_str = torch.version.cuda

    return Environment(
        os=collect_env.get_os(run_lambda),
        python_version=f'{sys_version} ({bit_count}-bit runtime)',
        python_platform=collect_env.get_python_platform(),
        pip_version=pip_version,
        pip_packages='\n'.join(pip_packages),
        torch_version=version_str,
        torch_is_debug=debug_mode_str,
        cpu_info=cpu_info,
        total_ram_gb=total_ram,
        cuda_is_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=collect_env.get_running_cuda_version(run_lambda),
        cuda_module_loading=collect_env.get_cuda_module_loading_config(),
        nvidia_gpu_models=collect_env.get_gpu_info(run_lambda),
        nvidia_driver=collect_env.get_nvidia_driver_version(run_lambda),
        cudnn_version=collect_env.get_cudnn_version(run_lambda),
    )


ENVIRONMENT_FORMAT = """
OS: {os}
CPU: {cpu_info}
RAM: {total_ram_gb} GB

Python version: {python_version}
Python platform: {python_platform}

Pip version: {pip_version}
Pip packages:
{pip_packages}

PyTorch version: {torch_version}
Is debug build: {torch_is_debug}
CUDA used to build PyTorch: {cuda_compiled_version}

Is CUDA available: {cuda_is_available}
CUDA runtime version: {cuda_runtime_version}
CUDA_MODULE_LOADING set to: {cuda_module_loading}
Nvidia driver version: {nvidia_driver}
cuDNN version: {cudnn_version}
GPU models and configuration:
{nvidia_gpu_models}
""".strip()


def log_environment(
    level: int = logging.INFO,
    ranks: Iterable[int] | None = (0,),
) -> None:
    """Log the hardware and software environment.

    Args:
        level: Logging level.
        ranks: Ranks to log the environment on. If `None`, logs on all ranks.
    """
    env = collect_environment()
    env_str = ENVIRONMENT_FORMAT.format(**env._asdict())
    logger.log(
        level,
        f'Runtime environment:\n{env_str}',
        extra={'ranks': ranks},
    )


if __name__ == '__main__':  # pragma: no cover
    env = collect_environment()
    env_str = ENVIRONMENT_FORMAT.format(**env._asdict())
    print(env_str)
