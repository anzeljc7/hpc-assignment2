#!/bin/bash

#SBATCH --reservation=fri
#SBATCH --partition=gpu
#SBATCH --job-name=lenia_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=lenia_gpu_out.log

#LOAD MODULES 
module load CUDA

#BUILD
make

#RUN
srun ./lenia.out

