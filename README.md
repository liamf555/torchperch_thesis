# Setup on BCp4

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

Please add anything else I've missed...
