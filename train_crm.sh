#!/bin/bash
#SBATCH --job-name crm
#SBATCH --gpus 1
#SBATCH --time 3-00:00:00
#SBATCH --nodelist a6k01


python train_crm.py \
    --lr 1e-5 \
    --save_file gsm8k_crm_k:4_soft_lr_1e-5_deberta_v3_large