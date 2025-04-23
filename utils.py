from statistics import mean
from torch.utils.data import Dataset
from collections import OrderedDict
import xml.etree.ElementTree as ET
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import re
import random
import time
import datetime
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm



feedback_prompt="Review your previous answer and find problems with your answer."
refine_prompt="Based on the problems you found, improve your answer. Please reiterate your answer, with your final answer a single numerical number, in the form \boxed{answer}."

dataset_id={
    'gsm8k': "openai/gsm8k",
    'coin_flip': "skrishna/coin_flip_2",
    'commonsenseqa': "tau/commonsense_qa",
    'strategyqa': 'ChilleD/StrategyQA'
}

def datasetLoader(dataset_name, split='train'):
    dataloader=[]
    if dataset_name == 'gsm8k':
        dataset = load_dataset(dataset_id[dataset_name], "main")
    else:
       dataset =  load_dataset(dataset_id[dataset_name])
    if split == 'train':
        dataset = dataset['train']
    else:
       dataset = dataset['test']
    for data in tqdm(dataset):
        if dataset_name == 'gsm8k':
            dataloader.append((data['question'], data['answer'].split("####")[-1].strip()))
        elif dataset_name == 'commonsenseqa':
            choices = "Answer Choices: "
            for idx in range(len(data['choices']['label'])):
                choices += f"({data['choices']['label'][idx]}) {data['choices']['text'][idx]} "
            dataloader.append((data['question']+choices, data['answerKey']))
        elif dataset_name == 'coin_flip':
           label = "Yes" if data['targets'] == "true" else "No"
           dataloader.append((data['inputs'].split("Q: ")[-1].strip(), data['targets']))
        elif dataset_name == 'strategyqa':
           label = "Yes" if data['answer'] == "true" else "No"
           dataloader.append((data['question'], label))
        else:
           raise NotImplementedError(f"{dataset} is currently not implemented")
    return dataloader
        

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

# generate data
def inference(chat, model, tokenizer, device, sampling=True, temperature=0.6):
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)
    output = model.generate(tokenized_chat, max_new_tokens=512, do_sample=sampling, temperature=temperature)
    decode_input = tokenizer.batch_decode(tokenized_chat, skip_special_tokens=True)[0]
    decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decode_output[len(decode_input):]

def inference_batch(chat, model, tokenizer, device, sampling=True, temperature=0.6):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    outputs=[]
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    tokenized_chat = tokenizer(tokenized_chat, padding=True, return_tensors='pt')
    tokenized_chat = tokenized_chat.to(device)
    output = model.generate(**tokenized_chat, max_new_tokens=512, do_sample=sampling, temperature=temperature)
    decode_input = tokenizer.batch_decode(tokenized_chat['input_ids'], skip_special_tokens=True)
    decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)
    for do,di in zip(decode_output, decode_input):
        outputs.append(do[len(di):])
    return outputs


def inference_w_ntg(args, chat, model, tokenizer, device):
    tokenized_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False, return_tensors="pt")
    # only for llama
    if tokenized_chat.endswith("<|eot_id|>"):
       tokenized_chat = tokenized_chat[:-len("<|eot_id|>")]
    tokenized_chat+=args.direct_answer_trigger
    tokenized_chat = tokenizer([tokenized_chat], return_tensors="pt")
    tokenized_chat = tokenized_chat.to(device)
    output = model.generate(**tokenized_chat, max_new_tokens=512)
    decode_input = tokenizer.batch_decode(tokenized_chat['input_ids'], skip_special_tokens=True)[0]
    decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decode_output[len(decode_input):]


def check(args, pred, label):
    try:
        pred = answer_cleansing(args, pred)    
        pred = clean_pred(pred)
        label = clean_ans(label)
        if pred == label:
            return True
    except:
        pass
    return False


def generate_data(idx_num, model, tokenizer, question, answer, args, device):
    insts=[]
    chat=[
        {"role": "user", "content": question}
    ]
    # with open("current.txt", "w") as f:
    #    text = f"{idx_num}. {question}\n\nlabel:{answer}"
    #    f.write(text+'\n')
    
    initial_response = inference(chat, model, tokenizer, device, sampling=False)
    print("initial_response: ", initial_response)
    # with open("current.txt", "a+") as f:
    #    text = f"initial_response: {initial_response}"
    #    f.write(text+'\n')

    chat.append({"role": "assistant", "content": initial_response})
    chat.append({"role": "user", "content": feedback_prompt})
    # TODO: need to save verbose
    for idx in range(args.num_feedbacks):
        revisions=[]
        correct=[]
        feedback = inference(chat, model, tokenizer, device)

        # with open("current.txt", "a+") as f:
        #     text = f"{idx+1} / {args.num_feedbacks} feedback: {feedback}"
        #     f.write(text+'\n\n')
        
        partial_chat=[
            {"role": "assistant", "content": feedback},
            {"role": "user", "content": refine_prompt}
        ]
        cnt=0
        print(f"feedback #{idx}: ", feedback)
        for i in range(args.num_revision):
            revision = inference(chat+partial_chat, model, tokenizer, device, sampling=True, temperature=1.0)
            print(revision)
            revisions.append(revision)
            # with open("current.txt", "a+") as f:
            #     text = f"{i+1} / {args.num_revision} revision: {revision}"
            #     f.write(text+'\n')
            extract_answer_chat =[
                {"role": "assistant", "content": revision}
            ]
            pred = inference_w_ntg(args, chat+partial_chat+extract_answer_chat, model, tokenizer, device)
            print(pred)
            
            if check(args, pred, answer):
                cnt+=1
                correct.append(True)
            else:
                correct.append(False)
            # with open("current.txt", "a+") as f:
            #     text = f"correct: {check(args, pred, answer)}"
            #     f.write(text+'\n\n')
        print(f"score for feedback #{idx}: ", cnt/args.num_revision*100)
        insts.append({
            'idx': idx_num,
            'question': question,
            'initial_response': initial_response,
            'feedback': feedback,
            'revised_responses': revisions,
            'correct': correct,
            'score': cnt / args.num_revision
        })
    return insts


    
# generate data end

def generate_data_batch(idx_num, model, tokenizer, question, answer, args, device):
    insts=[]
    chat=[
        {"role": "user", "content": question}
    ]
    # with open("current.txt", "w") as f:
    #    text = f"{idx_num}. {question}\n\nlabel:{answer}"
    #    f.write(text+'\n')
    
    initial_response = inference(chat, model, tokenizer, device, sampling=False)
    print("initial_response: ", initial_response)
    # with open("current.txt", "a+") as f:
    #    text = f"initial_response: {initial_response}"
    #    f.write(text+'\n')

    chat.append({"role": "assistant", "content": initial_response})
    chat.append({"role": "user", "content": feedback_prompt})

    # number of feedbacks * chat
    feedback_chat = [chat for _ in range(args.feedbacks)]
    revisions=[]
    correct = []
    feedbacks = inference_batch(feedback_chat, model, tokenizer, device)
    for feedback in feedbacks:
        partial_chat = [
            {"role": "assistant", "content": feedback},
            {"role": "user", "content": refine_prompt}
        ]
        refine_chat = [chat+partial_chat for _ in range(args.num_revision)]
        revisions = inference_batch(refine_chat, model, tokenizer, device, sampling=True, temperature=1.0)
        for rc, r in zip(refine_chat, revisions):
            rc.append({"role": "assistant", "content": r})
            rc.append({"role": "user", "content": args.direct_answer_trigger})
        final_predictions = inference_batch(refine_chat, model, tokenizer, device, sampling=False)
        for pred in final_predictions:
            if check(args, pred, answer):
                cnt+=1
                correct.append(True)
            else:
                correct.append(False)
        insts.append({
            'idx': idx_num,
            'question': question,
            'initial_response': initial_response,
            'feedback': feedback,
            'revised_response': revisions,
            'correct': correct,
            'score': cnt / args.num_revision
        })
    return insts





def shuffleDict(d):
  keys = list(d.keys())
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  [(key, d[key]) for key in keys]
  random.shuffle(keys)
  keys = [(key, d[key]) for key in keys]
  #keys = d(keys)
  return dict(keys)
  
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    now = now.strftime('%Y/%m/%d %H:%M:%S')
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass

def data_reader(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()
    if args.dataset == "aqua":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "(" + "(".join(json_res["options"])
          choice = choice.replace("(", " (").replace(")", ") ")
          choice = "Answer Choices:" + choice
          questions.append(json_res["question"].strip() + " " + choice)
          answers.append(json_res["correct"])
  
    elif args.dataset == "gsm8k":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          questions.append(json_res["question"].strip())
          answers.append(json_res["answer"].split("#### ")[-1])
  
    elif args.dataset == "commonsenseqa":
      with open(args.dataset_path) as f:
        lines = f.readlines()
        for line in lines:
          json_res = decoder.raw_decode(line)[0]
          choice = "Answer Choices:"
          for c in json_res["question"]["choices"]:
              choice += " ("
              choice += c["label"]
              choice += ") "
              choice += c["text"]
          questions.append(json_res["question"]["stem"].strip() + " " + choice)
          answers.append(json_res["answerKey"])

    elif args.dataset in ("addsub", "multiarith", "singleeq"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
          q = line["sQuestion"].strip()
          a = str(line["lSolutions"][0])
          if a[-2:] == ".0":
              a = a[:-2]
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "strategyqa":
      with open(args.dataset_path) as f:
        json_data = json.load(f)["examples"]
        for line in json_data:
          q = line["input"].strip()
          a = int(line["target_scores"]["Yes"])
          if a == 1:
              a = "yes"
          else:
              a = "no"
          questions.append(q)
          answers.append(a)
        
    elif args.dataset == "svamp":
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        for line in json_data:
            q = line["Body"].strip() + " " + line["Question"].strip()
            a = str(line["Answer"])
            if a[-2:] == ".0":
                a = a[:-2]
            questions.append(q)
            answers.append(a)
            
    elif args.dataset in ("bigbench_date", "object_tracking"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        if args.dataset == "bigbench_date":
            choice_index = ['A','B','C','D','E','F']
        elif args.dataset in ("object_tracking"):
            choice_index = ['A','B','C']
        else:
            raise ValueError("dataset is not properly defined ...")
        for line in json_data:
          q = line["input"].strip()
          if args.dataset == "bigbench_date":
              choice = "Answer Choices:"
              # Randomly shuffle the answer choice dictionary because the original answer is always A ...
              choice_dic = shuffleDict(line["target_scores"])
          elif args.dataset == "object_tracking":
              choice = "\nWhich choice is true ? Answer Choices:"
              choice_dic = line["target_scores"]
          else:
              raise ValueError("dataset is not properly defined ...")
          for i, key_value in enumerate(choice_dic.items()):
              key, value = key_value
              choice += " ("
              choice += choice_index[i]
              choice += ") "
              choice += key
              if value == 1:
                  a = choice_index[i]
                  #a = key
          q = q + " " + choice
          questions.append(q)
          answers.append(a)            
          
    elif args.dataset in ("coin_flip", "last_letters"):
      with open(args.dataset_path) as f:
        json_data = json.load(f)
        json_data = json_data["examples"]
        for line in json_data:
          q = line["question"]
          a = line["answer"]
          questions.append(q)
          answers.append(a)
        
    else:
        raise ValueError("dataset is not properly defined ...")
    
    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)
    
    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    
    return questions, answers

# Create dataset object before dataloader ...
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers = data_reader(args)
        self.len = len(self.questions)
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        return input, output

def setup_data_loader(args):
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)
    
    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, 3)
    print("dataloader_num_workers: " + str(dataloader_num_workers))
    
    dataset = MyDataset(args)
    
    dataloader = torch.utils.data.DataLoader(dataset,
                  shuffle=True,
                  batch_size=1,
                  drop_last=False,
                  num_workers=dataloader_num_workers,
                  worker_init_fn=seed_worker,
                  generator=g,
                  pin_memory=True)

    return dataloader

# ver 0.2
def answer_cleansing(args, pred):

    print("pred_before : " + pred)
    
    # if args.method in ("few_shot", "few_shot_cot"):
    #     preds = pred.split(args.direct_answer_trigger_for_fewshot)
    #     answer_flag = True if len(preds) > 1 else False 
    #     pred = preds[-1]

    if args.dataset in ("aqua", "commonsenseqa"):
        pred = re.findall(r'A|B|C|D|E', pred)
    elif args.dataset == "bigbench_date":
        pred = re.findall(r'A|B|C|D|E|F', pred)
    elif args.dataset in ("object_tracking"):
        pred = re.findall(r'A|B|C', pred)
    elif args.dataset in ("gsm8k", "addsub", "multiarith", "svamp", "singleeq"):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]
    elif args.dataset in ("strategyqa", "coin_flip"):
        pred = pred.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", pred)
        pred = pred.split(" ")
        pred = [i for i in pred if i in ("yes", "no")]
    elif args.dataset == "last_letters":
        right_index = pred.rfind('"')
        if right_index != -1:
            left_index = pred[:right_index].rfind('"')
            pred = pred[left_index:right_index+1].lower()
        pred = re.sub("\"|\'|\n|\.|\s","", pred)
        pred = [pred]
    else:
        raise ValueError("dataset is not properly defined ...")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list ...
                pred = pred[0]
            else:
                # choose the last element in list ...
                pred = pred[-1]
        elif args.method in ("zero_shot", "role_play"):
            # choose the first element in list ...
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ...")
    
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
    
    print("pred_after : " + pred)
    
    return pred