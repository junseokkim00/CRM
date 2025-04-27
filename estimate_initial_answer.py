import argparse
import torch
import random
import time
import os
from dotenv import load_dotenv
import json
from utils import fix_seed, print_now, feedback_prompt, refine_prompt, generate_data, datasetLoader, inference_w_ntg, check
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    print("load model and tokenizer")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, token=hf_token)
    model_name = args.model_name
    model_name = model_name.replace("/", "_")
    device = 'cuda'
    model.to(device)
    fix_seed(args.random_seed)

    print("setup data loader ...")

    dataloader = datasetLoader(args.dataset)

    # load dataset
    with open(args.data_path, "r") as f:
        lines = f.readlines()

    print_now()

    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    checklist = set()

    if args.end == 0:
        args.end = len(lines) // 3

    for i, line in enumerate(lines):
        idx = i // 3
        if args.start <= idx and idx < args.end:
            print('*************************')
            print("{}st data".format(i+1))

            # Prepare question template ...
            data = json.loads(line)
            if data['idx'] in checklist:
                continue

            checklist.add(data['idx'])
            x, y, response = data['question'], dataloader[idx][-1], data['initial_response']

            question = x
            answer = y.strip()
            

            chat = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ]

            pred = inference_w_ntg(args, chat, model, tokenizer, device)
            

            with open(f"./logs/{args.dataset}_{model_name}_initial_response.jsonl", "a+") as f:
                inst = {
                    'idx': idx,
                    'question': question,
                    'initial_response': response,
                    'pred': pred,
                    'answer': answer,
                    'correct': check(args, pred, answer)
                }
                f.write(json.dumps(inst)+'\n')


def clean_ans(ans):
    new_ans = ""
    for i in range(len(ans)):
        if ans[i] == ",":
            continue
        new_ans += ans[i]
    # print(ans, new_ans)

    if '.' in new_ans:
        pos = new_ans.find('.')
        if len(new_ans) - pos - 1 > 7:
            new_ans = new_ans[:pos + 7]
    return new_ans


def clean_pred(pred):
    if '.' in pred:
        pred = pred.rstrip('0')
        if pred.endswith('.'):
            pred = pred[:-1]

    if '.' in pred:
        pos = pred.find('.')
        if len(pred) - pos - 1 > 7:
            pred = pred[:pos + 7]
    return pred


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int,
                        default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsenseqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )

    parser.add_argument(
        "--data_path", type=str
    )

    parser.add_argument("--max_num_worker", type=int, default=3,
                        help="maximum number of workers for dataloader")

    # parser.add_argument(
    #     "--model", type=str, default="gpt3", choices=["turbo","gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    # )
    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct"
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot", "role_play"], help="method"
    )

    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )

    parser.add_argument(
        "--num_feedbacks", type=int, default=3
    )

    parser.add_argument(
        "--num_revision", type=int, default=4
    )

    parser.add_argument(
        "--start", type=int, default=0
    )

    parser.add_argument(
        "--end", type=int, default=0
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsenseqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the final answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
    return args


if __name__ == "__main__":
    main()
