#!/bin/bash
#SBATCH --job-name crm_correct_or_wrong
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00
#SBATCH --nodelist ac01


# Junseok Kim
# dataset=commonsenseqa

# Kyeongman Park
# dataset=strategyqa

# Kyochul Jang
# dataset=coin_flip




pairwise_data_path=logs/${dataset}_pairwise_correct.jsonl
model_name=meta-llama/Llama-3.2-3B-Instruct
output_dir=dpo_llama3b_output_w_correct_data_${dataset}

python train_dpo.py \
    --pairwise_data_path ${pairwise_data_path} \
    --model_name ${model_name} \
    --output_dir ${output_dir}


pairwise_data_path=logs/${dataset}_pairwise_wrong.jsonl
output_dir=dpo_llama3b_output_w_wrong_data_${dataset}

python train_dpo.py \
    --pairwise_data_path ${pairwise_data_path} \
    --model_name ${model_name} \
    --output_dir ${output_dir}