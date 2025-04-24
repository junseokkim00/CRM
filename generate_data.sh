#!/bin/bash
#SBATCH --job-name crm-generate-data
#SBATCH --gpus 1
#SBATCH --time 4-00:00:00
#SBATCH --nodelist ac01

# Junseok Kim
# python generate_data.py \
#     --dataset commonsenseqa \
#     --num_feedbacks 3 \
#     --num_revision 4

# Kyeongman Park
# python generate_data.py \
#     --dataset strategyqa \
#     --num_feedbacks 3 \
#     --num_revision 4

# Kyochul Jang
# python generate_data.py \
#     --dataset coin_flip \
#     --num_feedbacks 3 \
#     --num_revision 4