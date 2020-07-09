
# Torchperch

Torchperch is a numerical model of a custom, variable sweep aircraft used for reinforcment learning research, primarily for agile flight maneovures.

The aerodynamic parameters and equations of motion are contained in [bixler.py](../master/bixler/bixler.py). This script is augmented by the [scenarios](../master/scenarios/) and [controllers](../master/controllers/) to form an [OpenAI Gym environment](../master/gym-bixler/gym_bixler/envs/bixler_env.py). The primary focus is currently the perched landing manoeuvre using the sweep elevator controller. Recent work has looked to model [wind](../master/wind/).

## Installation

## Setup on BCp4

For BCp4 all you really need to do, once your environment is setup, is to run the `sbatch` command
with the job script as an argument:

    sbatch scripts/bc4/gym.job

You can view the status of the job with the `scripts/bc4/showstatus.sh` script (calls `sacct` with
some nicer formatting).

More docs on SLURM are [here](https://www.acrc.bris.ac.uk/protected/bc4-docs/index.html) (UoB only).

For testing, you can source the `environs/bluecrystal_p4-cuda-gym` file to setup the modules you need.
You will need an environment called "gym" with gym, stable-baselines and tensorflow installed:

    conda create -n gym python=3
    conda activate gym
	conda install tensorflow
	pip install gym
	pip install stable-baselines

The job script will use this environment to run the jobs so test that this is setup correctly first.
You can also submit jobs to the test queue to avoid using up resources.

