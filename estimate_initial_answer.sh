#!/bin/bash
#SBATCH --job-name estimate-initial-answer
#SBATCH --gpus 1
#SBATCH --time 4-00:00:00
#SBATCH --nodelist a6k01

# 1. generate_data.sh로 생성한 data의 filepath 아래에 추가하기 (.jsonl 파일)
data_path=logs/YOUR-DATA-PATH-FROM-generate_data.sh.jsonl


# 2. 자신이 맡은 dataset comment 해제후 실험 돌리기
# Junseok Kim
# dataset=commonsenseqa

# Kyeongman Park
# dataset=strategyqa

# Kyochul Jang
# dataset=coin_flip

python estimate_initial_answer.py \
    --dataset ${dataset} \
    --data_path ${data_path}