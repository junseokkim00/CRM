#!/bin/bash
#SBATCH --job-name prep-pairwise
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00


initial_response_path=logs/commonsenseqa_meta-llama_Llama-3.2-3B-Instruct_initial_response.jsonl
crm_data_path=logs/commonsenseqa_meta-llama_Llama-3.2-3B-Instruct_crm_k:3_n:4_start:0_end:9741.jsonl
dataset=commonsenseqa

python prepare_pairwise_data.py \
    --initial_response_path ${initial_response_path} \
    --crm_data_path ${crm_data_path} \
    --dataset ${dataset}