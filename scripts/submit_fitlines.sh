#!/bin/bash
#SBATCH --job-name=sedfit
#SBATCH --cpus-per-task=1
#SBATCH -t 0-0:10:00
#SBATCH --mem-per-cpu=1700M

module load miniconda
conda activate vgrace
python do_fitlines