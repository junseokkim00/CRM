#!/bin/bash
#SBATCH --job-name estimate-initial-answer
#SBATCH --gpus 1
#SBATCH --time 4-00:00:00
#SBATCH --nodelist a6k01

# 1. generate_data.sh로 생성한 data의 filepath 아래에 추가하기 (.jsonl 파일)
# data_path=logs/commonsenseqa_meta-llama_Llama-3.2-3B-Instruct_crm_k:3_n:4_start:0_end:9741.jsonl


# 2. 자신이 맡은 dataset comment 해제후 실험 돌리기
# Junseok Kim
dataset=commonsenseqa

# Kyeongman Park
# dataset=strategyqa

# Kyochul Jang
# dataset=coin_flip

# python estimate_initial_answer.py \
#     --dataset ${dataset} \
#     --data_path ${data_path}



# 8. DPO model를 이용해서 결과를 낸 값의 initial response의 결과값 확인하기
# python estimate_initial_answer_for_test.py \
#     --dataset ${dataset} \
#     --data_path logs/evaluate_commonsenseqa.jsonl