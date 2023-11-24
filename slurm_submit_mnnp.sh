#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --job-name=modern-nn-potentials
#SBATCH --mail-user=

#module load miniconda3

source activate modern-nn-potentials

cd ~/projects/modern_nn_potentials/scripts
python ./posteriors_script.py


