#!/bin/bash

#SBATCH --job-name=bixler_perching
#SBATCH --partition=gpu_veryshort
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type END
#SBATCH --mail-user robert.clarke@bristol.ac.uk

#SBATCH --time=0:0:30

#SBATCH -o ./output/%j.out

CODE_ROOT=${HOME}/torchperch

# Change to workdir and print
cd $SLURM_SUBMIT_DIR
echo "Current working_directory: $PWD"

# Setup the environment for running training
source environs/bluecrystal_p4-cuda-gym
conda activate gym
echo "Activated environment"

SCENARIO="${SCENARIO-perching}"
CONTROLLER="${CONTROLLER-sweep_elevator}"

# Echo some meta information
echo "Scenario: $SCENARIO"
echo "Controller: $CONTROLLER"
echo "Commit: $(git rev-parse HEAD)"
echo "JobID: $SLURM_JOBID"

echo "Beginning training"

JOB_NID=${SLURM_JOBID}-${SLURM_ARRAY_TASK_ID}

exec python ${CODE_ROOT}/gym_test.py
