#!/bin/bash
#SBATCH --job-name crm-eval
#SBATCH --gpus 2
#SBATCH --time 2-00:00:00
#SBATCH --nodelist a6k01

python evaluate_crm.py \
    --reference_model meta-llama/Llama-3.2-3B-Instruct \
    --dataset commonsenseqa \
    --batch_size 4

python evaluate_crm.py \
    --reference_model meta-llama/Llama-3.2-3B-Instruct \
    --dpo_model_path dpo_llama3b_output_w_correct_data_commonsenseqa/checkpoint-3804 \
    --dataset commonsenseqa \
    --batch_size 4 \
    --etc _dpo_correct


python evaluate_crm.py \
    --reference_model meta-llama/Llama-3.2-3B-Instruct \
    --dpo_model_path dpo_llama3b_output_w_full_data_commonsenseqa/checkpoint-5718 \
    --dataset commonsenseqa \
    --batch_size 4 \
    --etc _dpo_full

python evaluate_crm.py \
    --reference_model meta-llama/Llama-3.2-3B-Instruct \
    --dpo_model_path dpo_llama3b_output_w_wrong_data_commonsenseqa/checkpoint-1911 \
    --dataset commonsenseqa \
    --batch_size 4 \
    --etc _dpo_wrong