#!/bin/bash
#SBATCH --job-name=fitlines
#SBATCH --cpus-per-task=1
#SBATCH -t 0-4:00:00
#SBATCH --mem-per-cpu=3G


module load miniconda
conda activate vgrace
python do_fitlines.py -d /home/ek757/surveys/SAGAbg/local_data/ 