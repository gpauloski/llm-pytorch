# BERT Pretraining

BERT pretraining based on [NVIDIA's configuration](https://github.com/NVIDIA/DeepLearningExamples/blob/ca5ae20e3d1af3464159754f758768052c41c607/PyTorch/LanguageModeling/BERT/scripts/configs/pretrain_config.sh).

This guide assumes you have installed the `llm` packages and its dependencies as described in the [README](../README.md).

An example training configuration is provided in [configs/bert-large-uncased-pretraining.py](../configs/bert-large-uncased-pretraining.py).
Typically, you will need to change at least the `DATA_DIR`, `OUTPUT_DIR`, and `PHASE` depending on your training state.

## Run Commands

- **Single-GPU for Debugging:**
  ```bash
  python -m llm.trainers.bert --config configs/bert-large-uncased-pretraining.py --debug
  ```
- **Multi-GPU Single-Node:**
  ```bash
  torchrun --nnodes=1 --nproc_per_node=auto --standalone \
      -m llm.trainers.bert --config configs/bert-large-uncased-pretraining.py
  ```
- **Multi-Node Multi-GPU:**
  ```bash
  torchrun --nnodes $NNODES --nproc_per_node=auto --max_restarts 0 \
      --rdzv_backend=c10d --rdzv_endpoint=$PRIMARY_RANK \
      -m llm.trainers.bert --config configs/bert-large-uncased-pretraining.py
  ```

### PBS Training

This is an example submission script for a PBS scheduler.
```bash
#!/bin/bash
#PBS -A __ALLOCATION__
#PBS -q __QUEUE__
#PBS -M __EMAIL__
#PBS -m e
#PBS -l select=16:system=polaris
#PBS -l walltime=6:00:00
#PBS -l filesystems=home:grand
#PBS -o pbs-logs
#PBS -j eo

# Figure out training environment based on PBS_NODEFILE existence
if [[ -z "${PBS_NODEFILE}" ]]; then
    RANKS=$HOSTNAME
    NNODES=1
else
    PRIMARY_RANK=$(head -n 1 $PBS_NODEFILE)
    RANKS=$(tr '\n' ' ' < $PBS_NODEFILE)
    NNODES=$(< $PBS_NODEFILE wc -l)
fi

CONFIG="configs/bert-large-uncased-pretraining.py"

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
    LAUNCHER+="--rdzv_backend=c10d --rdzv_endpoint=$PRIMARY_RANK "
fi

# Training script and parameters
CMD=" $LAUNCHER -m llm.trainers.bert --config $CONFIG "
echo "Training Command: $CMD"

mpiexec --hostfile $PBS_NODEFILE -np $NNODES --env OMP_NUM_THREADS=8 --cpu-bind none $CMD
```

## Transitioning from Phase 1 to 2

After phase 1 training is complete, set `PHASE = 2` in the config file.
Then, copy the last checkpoint from the phase 1 directory to a new phase 2 checkpoint directory.

*Note: checkpointing is still incomplete.*
