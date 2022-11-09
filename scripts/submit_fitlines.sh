#!/bin/bash
#SBATCH --job-name=SCAVENGE_fitlines
#SBATCH --cpus-per-task=1
#SBATCH -t 0-2:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=scavenge
#SBATCH --requeue


python do_fitlines.py -S 300 -E 400 -d ../local_data/
