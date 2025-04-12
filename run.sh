#!/bin/bash
#SBATCH --job-name generate-crm-data
#SBATCH --gpus 1
#SBATCH --time 1-00:00:00
#SBATCH --nodelist a6k01

# Junseok Kim
# python generate_data.py \
#     --dataset gsm8k \
#     --num_feedbacks 3 \
#     --num_revision 4 \
#     --start 0 \
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