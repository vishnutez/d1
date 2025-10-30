#!/bin/bash
#SBATCH --job-name=get_acc_math
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=8 # number of CPUs per node
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --output=logs_get_acc/%j.out

ml Miniconda3
ml WebProxy
source activate /scratch/user/vishnukunde/.conda/envs/d1

python parse_and_get_acc.py