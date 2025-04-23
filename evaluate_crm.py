# bring reference model and dpo trained model and check whether it choose a better output
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from utils import feedback_prompt, refine_prompt
from tqdm import tqdm
import json
import re


dataset_id={
    'gsm8k': "openai/gsm8k",
    'coin_flip': "skrishna/coin_flip_2",
    'commonsenseqa': "tau/commonsense_qa",
    'strategyqa': 'ChilleD/StrategyQA'
}

direct_answer_trigger = {
    'gsm8k': "Therefore, the answer (arabic numerals) is",
    'coin_flip': "Therefore, the answer (Yes or No) is",
    'commonsenseqa': "Therefore, among A through E, the answer is",
    'strategyqa': "Therefore, the answer (Yes or No) is"
}

class testDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    def __getitem__(self, index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

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

def check(pred, label):
    pred = pred.replace(',','')
    label = label.replace(',','')
    pred = re.findall(r"\d+", pred)
    if len(pred) > 0:
        pred = pred[-1]
    else:
        pred = ""
    return pred == label


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

def self_refine(args, batch, ref_model, dpo_model, tokenizer, device):
    chats=[]
    for question in batch[0]:
        chats.append([
            {"role": "user", "content": question}
        ])
    # 2. generate initial_response
    initial_responses = inference_batch(chats, ref_model, tokenizer, device, sampling=False)
    for initial_response, chat in zip(initial_responses, chats):
        chat.append(
            {"role": "assistant", "content": initial_response}
        )
        chat.append(
            {"role": "user", "content": feedback_prompt}
        )
    # 3. generate feedback for each response
    if dpo_model is None:
        feedbacks = inference_batch(chats, ref_model, tokenizer, device, sampling=True)
    else:
        feedbacks = inference_batch(chats, dpo_model, tokenizer, dpo_model.device, sampling=True)
    for feedback, chat in zip(feedbacks, chats):
        chat.append(
            {"role": "assistant", "content": feedback}
        )
        chat.append(
            {"role": "user", "content": refine_prompt}
        )

    # 4. generate final response
    final_responses = inference_batch(chats, ref_model, tokenizer, device, sampling=False)
    for final_response, chat in zip(final_responses, chats):
        chat.append(
            {"role": "assistant", "content": final_response}
        )
        chat.append(
            {"role": "user", "content": direct_answer_trigger[args.dataset]}
        )
    answers = inference_batch(chats, ref_model, tokenizer, device, sampling=False)
    return answers, initial_responses, feedbacks, final_responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--dpo_model_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=['gsm8k', 'coin_flip', 'commonsenseqa', 'strategyqa'], default='gsm8k')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--etc", type=str, default='')
    args = parser.parse_args()

    dataset = datasetLoader(args.dataset, split='test')
    dataset = testDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.reference_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.reference_model)
    device='cuda:0'
    ref_model.to(device)
    if args.dpo_model_path is None:
        dpo_model = None
    else:
        dpo_model = AutoModelForCausalLM.from_pretrained(args.dpo_model_path)
        device2='cuda:1'
        dpo_model.to(device2)
    
    cnt=0
    for batch in tqdm(dataloader):
        labels = batch[1]
        answers, initial_responses, feedbacks, final_responses = self_refine(args, batch, ref_model, dpo_model, tokenizer, device)
        with open(f"./logs/evaluate_{args.dataset}{args.etc}.jsonl", "a+") as f:
            for i in range(len(answers)):
                inst = {
                    'idx': cnt,
                    'question': batch[0][i],
                    'initial_response': initial_responses[i],
                    'feedback': feedbacks[i],
                    'final_response': final_responses[i],
                    'pred': answers[i],
                    'label': labels[i]
                }
                f.write(json.dumps(inst)+'\n')
    


