import argparse
import torch
import random
import time
import os
from dotenv import load_dotenv
import json
from utils import fix_seed, print_now, feedback_prompt, refine_prompt, generate_data, datasetLoader
from transformers import AutoTokenizer, AutoModelForCausalLM




dataset_id={
    'gsm8k': "openai/gsm8k",
    'coin_flip': "skrishna/coin_flip_2",
    'commonsenseqa': "tau/commonsense_qa",
    'strategyqa': 'ChilleD/StrategyQA'
}


def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    print("load model and tokenizer")
    load_dotenv()
    hf_token=os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, token=hf_token)
    model_name = args.model_name
    model_name = model_name.replace("/","_")
    device='cuda'
    model.to(device)
    fix_seed(args.random_seed)
    
    
    print("setup data loader ...")

    # load dataset
    dataloader = datasetLoader(args.dataset)
    
    print_now()


    if not os.path.exists('./logs'):
        os.mkdir('./logs')
            
    for i, data in enumerate(dataloader):
        if args.start <= i and i < args.end:
            print('*************************')
            print("{}st data".format(i+1))
                    
            # Prepare question template ...
            x, y = data

            question = x
            answer = y.strip()

            insts = generate_data(i, model, tokenizer, question, answer, args, device)

            with open(f"./logs/{args.dataset}_{model_name}_crm_k:{args.num_feedbacks}_n:{args.num_revision}_start:{args.start}_end:{args.end}.jsonl", "a+") as f:
                for inst in insts:
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
        if len(new_ans) - pos -1 > 7:
            new_ans = new_ans[:pos + 7]
    return new_ans
    
def clean_pred(pred):
    if '.' in pred:
        pred = pred.rstrip('0')
        if pred.endswith('.'):
            pred = pred[:-1]

    if '.' in pred:
        pos = pred.find('.')
        if len(pred) - pos -1 > 7:
            pred = pred[:pos + 7]
    return pred
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    
    parser.add_argument(
        "--dataset", type=str, default="aqua", choices=["aqua", "gsm8k", "commonsenseqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "bigbench_date", "object_tracking", "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    
    parser.add_argument("--max_num_worker", type=int, default=3, help="maximum number of workers for dataloader")
    
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
        "--limit_dataset_size", type=int, default=10, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
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