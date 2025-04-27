import os
import json
import argparse
from typing import Dict
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    HfArgumentParser,
)
from trl import DPOTrainer, DPOConfig

# 1. 데이터셋 처리


def preprocess_dpo_dataset(jsonl_path: str, tokenizer) -> Dataset:
    feedback_prompt = "Review your previous answer and find problems with your answer."
    examples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)

            # 시스템 메시지는 선택사항 (없어도 괜찮음)
            messages = [
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": entry["initial_response"]},
                {"role": "user", "content": feedback_prompt},
                {"role": "assitant", "content": ""}
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            examples.append({
                "prompt": prompt,
                "chosen": entry["feedback_w"],
                "rejected": entry["feedback_l"]
            })

            # print(examples[-1])
            # input()

    return Dataset.from_list(examples)

# 2. Train 스크립트


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairwise_data_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    # config
    # json_path = "./logs/commonsenseqa_pairwise_correct.jsonl"  # <- 수정 필요
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # output_dir = "./dpo_llama3b_output_w_correct_data_commonsenseqa"
    json_path = args.pairwise_data_path
    model_name = args.model_name
    output_dir = args.output_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto")
    dataset = preprocess_dpo_dataset(json_path, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    # DPO 설정
    training_args = DPOConfig(
        beta=0.1,
        max_prompt_length=512,
        max_length=1024,
        label_pad_token_id=-100,
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        bf16=True,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
