# BERT Pretraining

This guide walks through BERT pretraining based on [NVIDIA's configuration](https://github.com/NVIDIA/DeepLearningExamples/blob/ca5ae20e3d1af3464159754f758768052c41c607/PyTorch/LanguageModeling/BERT/scripts/configs/pretrain_config.sh){target=_blank}.
This configuration uses large batch training with LAMB to achieve 64K phase 1 and 32K phase 2 batch sizes.


## Setup

This guide assumes you have installed the `llm` packages and its dependencies as described in the [Installation Guide](../installation/index.md).
You also need the BERT pretraining dataset from [NVIDIA's examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#getting-the-data){target=_blank}.

We will use the configuration is provided in [configs/bert-large-lamb.py](https://github.com/gpauloski/llm-pytorch/blob/main/configs/bert-large-lamb.py){target=_blank}.
We start by copying the example configuration into our training directory.
Note all of the commands are from the root of the `llm-pytorch` directory.
```bash
mkdir -p runs/bert-large-pretraining/
cp configs/bert-large-lamb.py runs/bert-large-pretraining/config.py
```
You can inspect `runs/bert-large-pretraining/config.py` to see if you want
to adjust any options, though the default paths will work.

## Running the Trainer

There are a number of ways you may launch the trainer:

- **Single-GPU for Debugging:**
  ```bash
  python -m llm.trainers.bert --config path/to/config.py --debug
  ```
- **Multi-GPU Single-Node:**
  ```bash
  torchrun --nnodes=1 --nproc_per_node=auto --standalone \
      -m llm.trainers.bert --config path/to/config.py
  ```
- **Multi-Node Multi-GPU:**
  ```bash
  torchrun --nnodes $NNODES --nproc_per_node=auto --max_restarts 0 \
      --rdzv_backend=c10d --rdzv_endpoint=$PRIMARY_RANK \
      -m llm.trainers.bert --config path/to/config.py
  ```

Typically, you will want to use the last option inside of the script you submit
to your batch scheduling system.
This is an example submission script for a PBS scheduler.
```bash
#!/bin/bash
#PBS -A __ALLOCATION__
#PBS -q __QUEUE__
#PBS -M __EMAIL__
#PBS -m abe
#PBS -l select=16:system=polaris
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:grand
#PBS -j oe

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

CONFIG="runs/bert-large-pretraining/config.py"

# Commands to run prior to the Python script for setting up the environment
module load cray-python
module load cudatoolkit-standalone/11.7.1
source /path/to/virtualenv

# torchrun launch configuration
LAUNCHER="torchrun "
LAUNCHER+="--nnodes=$NNODES --nproc_per_node=auto --max_restarts 0 "
if [[ "$NNODES" -eq 1 ]]; then
    LAUNCHER+="--standalone "
else
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$PRIMARY_RANK"
fi

# Training script and parameters
CMD="$LAUNCHER -m llm.trainers.bert --config $CONFIG"
echo "Training Command: $CMD"

mpiexec --hostfile $PBS_NODEFILE -np $NNODES --env OMP_NUM_THREADS=8 --cpu-bind none $CMD
```

## Training Phase 1 and 2

Train the model for phase 1. After the end of phase 1, you'll see a checkpoint
named `runs/bert-large-pretraining/checkpoints/phase-1/global_step_7039.pt`.

To transition to phase 2, set `PHASE = 2` in the config file.
Then create a new directory for phase 2 checkpoints at
`runs/bert-large-pretraining/checkpoints/phase-2`.
Copy the last checkpoint from phase 1 to the phase 2 directory with the name `global_step_0.pt`.
Continue training to complete phase 2/

## Converting the Pretrained Model

TODO

## SQUAD Evaluation

TODO
