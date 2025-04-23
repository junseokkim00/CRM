#!/bin/bash
#SBATCH --job-name crm
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --nodelist ac01


python train_dpo.py