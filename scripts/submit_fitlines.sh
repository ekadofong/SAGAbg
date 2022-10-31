#!/bin/bash
#SBATCH --job-name=sedfit
#SBATCH --cpus-per-task=1
#SBATCH -t 0-0:10:00
#SBATCH --mem-per-cpu=1700M

module load miniconda
conda activate vgrace
python do_fitlines.py -d /home/ek757/surveys/SAGAbg/local_data/ -S 0 -E 10 