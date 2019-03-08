#!/bin/bash

#SBATCH --job-name=bixler_perching
#SBATCH --partition=gpu_veryshort
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type END
#SBATCH --mail-user robert.clarke@bristol.ac.uk

#SBATCH --time=0:0:30

# Does not work... :(
#SBATCH --output /mnt/storage/home/rc13011/torchperch/output/%A-%a/stdout.log
#SBATCH --error  /mnt/storage/home/rc13011/torchperch/output/%A-%a/stderr.log

# Change to workdir and print
cd $SLURM_SUBMIT_DIR
echo "Current working_directory: $PWD"

# Setup the environment for running training and notify
source environs/bluecrystal_p4-cuda
conda activate pytorch
echo "Activated environment"

SCENARIO="${SCENARIO-cobra}"
CONTROLLER="${CONTROLLER-sweep_elevator}"

# Echo some meta information
echo "Scenario: $SCENARIO"
echo "Controller: $CONTROLLER"
echo "Commit: $(git rev-parse HEAD)"
echo "JobID $PBS_JOBID"

echo "Beginning training"

JOB_NID=${SLURM_JOBID}-${SLURM_ARRAY_TASK_ID}
mkdir output/$JOB_NID

exec python learn.py --no-stdout \
    --scenario $SCENARIO \
    --controller $CONTROLLER \
    --logfile output/$JOB_NID/learning_log.txt \
    --networks output/$JOB_NID/networks
