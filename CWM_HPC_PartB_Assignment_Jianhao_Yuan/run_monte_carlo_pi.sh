#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --job-name=vectoradd
#SBATCH --gres=gpu:1

module purge
module load gpu/cuda

./monte_carlo_pi
