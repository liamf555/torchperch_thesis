#!/bin/bash

#PBS -N bixler_perching
#PBS -l nodes=1:ppn=8
#PBS -l walltime=0:02:00

# Change to workdir and print
cd $PBS_O_WORKDIR
echo "Current working_directory: $PWD"

# Get the environment for running training and notify
source environment_blueP3
conda activate pytorch
echo "Activated environment"

SCENARIO="${SCENARIO-perching}"
CONTROLLER="${CONTROLLER-sweep_elevator}"

# Echo some meta information
echo "Scenario: $SCENARIO"
echo "Controller: $CONTROLLER"
echo "Commit: $(git rev-parse HEAD)"
echo "JobID $PBS_JOBID"

echo "Beginning training"
exec python learn.py --no-stdout \
    --scenario $SCENARIO \
    --controller $CONTROLLER \
    --logfile learning_$PBS_JOBID.txt \
    --networks networks_$PBS_JOBID
