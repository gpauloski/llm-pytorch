# GPT Pretraining

This guide walks you through pretraining GPT-like causal language model.
Instructions are specific to ALCF's Polaris machine; however, the general
steps should apply for any system.

The training script is based on
[HuggingFace's CLM example](https://github.com/huggingface/transformers/blob/edf77728268e22e18151abb3d8acbb50ad8e92a8/examples/pytorch/language-modeling/run_clm_no_trainer.py){target=_blank}.

## Installation

1. Clone the repository.
   ```bash
   $ git clone https://github.com/gpauloski/llm-pytorch
   $ cd llm-pytorch
   ```
2. Load the Python and CUDA modules on Polaris. These modules will need to be
   loaded each time you activate the virtual environment.
   ```bash
   $ module load cray-python/3.9.12.1
   $ module load cudatoolkit-standalone/11.7.1
   ```
3. Create a virtual environment.
   ```bash
   $ python -m venv venv
   $ . venv/bin/activate
   ```
4. Install PyTorch and the `llm-pytorch` package. Other versions of PyTorch
   should work fine. I have personally tests PyTorch 1.13.1 with CUDA 11.7
   on Polaris.
   ```bash
   $ pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
   $ pip install -e .[kfac]
   ```

## Running the Scripts

The basic training command is `python -m llm.trainers.gpt {options}`.
The script uses [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index){target=_blank}
to detect the distributed environment.
I suggest using [`torchrun`](https://pytorch.org/docs/stable/elastic/run.html){index=_blank}
for distributed training. E.g.,

```bash
torchrun --nnodes=1 --nproc_per_node=auto --standalone -m llm.trainers.gpt {options}
```

Here is an example job script for Polaris (using PBS) which will automatically
set up the distributed environment according to your job parameters.
Note that this example trains a small 125M parameter GPT model on WikiText.
Highlighted lines contain information that you must complete yourself.

```bash title="pretrain.pbs" linenums="1" hl_lines="2 3 4 6 7 43"
#!/bin/bash
#PBS -A __ALLOCATION__
#PBS -q __QUEUE__
#PBS -M __EMAIL__
#PBS -m abe
#PBS -l select=16:system=polaris
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:grand
#PBS -j oe

# This stores the list of command line arguments passed to llm.trainers.gpt.
# WARNING: this is not an exhaustive list of options. See:
#     python -m llm.trainers.gpt --help
OPTIONS="--low_cpu_mem_usage "

# Dataset options
OPTIONS+="--dataset_name wikitext "
OPTIONS+="--dataset_config_name wikitext-2-raw-v1 "

# Model options
OPTIONS+="--model_name_or_path EleutherAI/gpt-neo-125m "

# Logging/checkpointing options
OPTIONS+="--output_dir runs/gpt-neo-125m-pretraining "
OPTIONS+="--checkpointing_steps 1000 "
OPTIONS+="--resume_from_checkpoint "

# Training parameters
OPTIONS+="--max_train_steps 10000 "
OPTIONS+="--per_device_train_batch_size 1 "
OPTIONS+="--per_device_eval_batch_size 1 "
OPTIONS+="--gradient_accumulation_steps 8 "
OPTIONS+="--mixed_precision fp16 "

# KFAC options
# OPTIONS+="--kfac "
# OPTIONS+="--kfac-factor-update-steps 5 "
# OPTIONS+="--kfac-inv-update-steps 50 "

# Commands to run prior to the Python script for setting up the environment
module load cray-python
module load cudatoolkit-standalone/11.7.1
source /path/to/virtualenv

# Figure out training environment based on PBS_NODEFILE existence
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    PRIMARY_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
    cat $PBS_NODEFILE
fi

# torchrun launch configuration
LAUNCHER="torchrun "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$PRIMARY_RANK"
fi

# Training script and parameters
CMD="$LAUNCHER -m llm.trainers.gpt $OPTIONS"
echo "Training Command: $CMD"

mpiexec --hostfile $PBS_NODEFILE -np $NNODES --env OMP_NUM_THREADS=8 --cpu-bind none $CMD
```

After updating `pretrain.pbs` accordingly, you can either execute the script
directly in an interactive session or submit it as a batch job.

**Interactive:**
```bash
$ qsub -A {ALLOC} -l select=1:system=polaris -l walltime=1:00:00 -I -q debug -l filesystems=home:grand
$ chmod u+x pretrain.pbs
$ ./pretrain.pbs
```
**Batch:**
```bash
$ qsub pretrain.pbs
```

## Customize Pretraining

### Model

This script uses HuggingFace models. The argument `--model_name_or_path` takes
either a path to a saved HuggingFace model directory or the name of a model on
the [HuggingFace Hub](https://huggingface.co/models){target=_blank}.

Here's some useful options:

* `--model_name_or_path EleutherAI/gpt-neo-125m`: Small GPT model useful for debugging.
* `--model_name_or_path EleutherAI/gpt-neo-1.3B`: Works with K-FAC and is almost the same size as GPT-2.
* `--model_name_or_path EleutherAI/gpt-neox-20b`: GPT NeoX 20B.
* `--model_name_or_path gpt2`: HuggingFace's GPT-2 implementation which uses Conv1D layers instead of Linear layers so does not work with K-FAC.

Note that the `--low_cpu_mem_usage` option will instantiate the model
architecture for pretraining without downloading the actual weights.

Alternatively, a `--config_name` and `--tokenizer_name` can be provided where
each can either be a name of an existing model/tokenizer or a path to their
respective HuggingFace compatible configurations.

### Dataset

There are two ways to provide a pretraining dataset: via
[HuggingFace Datasets](https://huggingface.co/docs/datasets/index){target=_blank}
or CSV/JSON/text files.

To use an existing dataset via the [Dataset Hub](https://huggingface.co/datasets){target=_blank},
find the name of the dataset and the name of the subset.
```bash
# Generic format
--dataset_name {name} --dataset_config_name {subset}
# WikiText (good for testing)
--dataset_name wikitext --dataset_config_name wikitext-2-raw-v1
# The Pile
--dataset_name EleutherAI/pile --dataset_config_name all
# The Pile-10K (subset for testing)
--dataset_name NeelNanda/pile-10k
```

Datasets are downloaded to `~/.cache/huggingface/datasets`.
This can be changed by setting `HF_DATASETS_CACHE`.
```bash
$ export HF_DATASETS_CACHE="/path/to/another/directory"
```

### Checkpointing

Checkpointing is not enabled by default. Use `--checkpointing_steps {STEPS}`
to enable checkpointing. To resume training from a checkpoint, add
`--resume_from_checkpoint`.

## Limitations

* FP16 training with HuggingFace Accelerate is faster than FP32 but still
  uses the same amount of memory.
