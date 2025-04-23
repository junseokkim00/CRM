#!/bin/bash
#SBATCH --job-name crm-eval
#SBATCH --gpus 2
#SBATCH --time 2-00:00:00
#SBATCH --nodelist a6k01

# python evaluate_crm.py \
#     --reference_model meta-llama/Llama-3.2-3B-Instruct \
#     --dataset gsm8k \
#     --batch_size 4

python evaluate_crm.py \
    --reference_model meta-llama/Llama-3.2-3B-Instruct \
    --dpo_model_path dpo_llama3b_output_w_correct_data/checkpoint-5300 \
    --dataset gsm8k \
    --batch_size 4 \
    --etc _dpo_correct