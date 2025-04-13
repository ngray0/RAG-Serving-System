#!/bin/bash
echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"


# ===================
# Environment setup
# ===================

echo "Setting up bash enviroment"

# Make available all commands on $PATH as on headnode
source ~/.bashrc

# Make script bail out after first error
set -e

SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}


CONDA_ENV_NAME=MLSystems
echo "Activating conda environment: ${CONDA_ENV_NAME}"
conda activate ${CONDA_ENV_NAME}

echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"

# input data directory path on the DFS - change line below if loc different
data_path=/home/${USER}
rsync --archive --update --compress --progress ${data_path}/MLS-G31 ${SCRATCH_HOME}




COMMAND="python -u task_Ann_cpu2.py"
echo "Running provided command: ${COMMAND}"
eval "${COMMAND}"
echo "Command ran successfully!"

rm -rf ${SCRATCH_HOME}

echo "Moving output data back to DFS"

# =========================
# Post experiment logging
# =========================
echo ""
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"
