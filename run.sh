#!/bin/bash
#SBATCH --job-name crm-initial-data-verdict
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00
#SBATCH --nodelist a5k01

# Junseok Kim
# python generate_data.py \
#     --dataset gsm8k \
#     --num_feedbacks 3 \
#     --num_revision 4 \
#     --start 2355 \
#     --end 2491

# KyoChul Jang
# python generate_data.py \
#     --dataset gsm8k \
#     --num_feedbacks 3 \
#     --num_revision 4 \
#     --start 2491 \
#     --end 4982


# Kyeongman Park
# python generate_data.py \
#     --dataset gsm8k \
#     --num_feedbacks 3 \
#     --num_revision 4 \
#     --start 4982 \
#     --end 7473


# python estimate_initial_answer.py \
#     --dataset gsm8k \
#     --start 0 \
#     --end 7473 \
#     --data_path logs/gsm8k_full.jsonl

python estimate_initial_answer_for_test.py \
    --dataset gsm8k \
    --start 0 \
    --end 7473 \
    --data_path logs/evaluate_gsm8k.jsonl