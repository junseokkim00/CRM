#!/bin/bash
#SBATCH --job-name prep-pairwise
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00


initial_response_path=DATA-FROM-estimate_initial_answer.sh
crm_data_path=DATA-FROM-generate_data.sh
dataset=your-data

python prepare_pairwise_data.py \
    --initial_response_path ${initial_response_path} \
    --crm_data_path ${crm_data_path} \
    --dataset ${dataset}