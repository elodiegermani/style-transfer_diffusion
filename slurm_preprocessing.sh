#!/bin/bash
#SBATCH --job-name=preprocessing-left-hand # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-dev 
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=16           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=2:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=preprocessing-left-hand%j.out # output file name
#SBATCH --error=preprocessing-left-hand%j.err  # error file name

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u \
	/gpfswork/rech/gft/umh25bv/style-transfer_diffusion/run_preprocessing.py