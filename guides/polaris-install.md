# Polaris Installation Guide

1. Load Polaris modules.
   ```bash
   module load cray-python/3.9.12.1
   module load cudatoolkit-standalone/11.7.1
   ```
2. Clone and create your virtual environment.
   ```bash
   git clone git@github.com:gpauloski/llm-pytorch
   cd llm-pytorch
   ```
3. Create your virtual environment.
   ```bash
   python -m venv venv
   . venv/bin/activate
   ```
   **Note:** anytime you want to use the virtual environment you will need to
   load the above module and activate the virtual environment. It may be
   helpful to add these to your `~/.bashrc` or PBS job scripts.
3. Install PyTorch.
   ```bash
   pip install torch==1.13.1
   ```
4. Install the `llm` package.
   ```bash
   pip install -e .  # Use the [dev] extras install for developing
   ```
5. Install NVIDIA Apex.
   This must be done from a compute node.
   E.g., `qsub -A [ALLOCATION] -l select=1:system=polaris -l walltime=1:00:00 -I -q debug -l filesystems=home:grand`.
   The `apex` repository can be cloned anywhere.
   ```bash
   # Activate your modules and virtual environment
   module load cray-python/3.9.12.1 cudatoolkit-standalone/11.7.1
   . /path/to/venv/bin/activate

   # Install and load extra requirements
   pip install packaging
   module load gcc

   # Clone and install
   git clone git@github.com:NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
   ```
6. Done!
