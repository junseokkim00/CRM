import argparse
import json
from collections import defaultdict


def cmp_func(idx, question, initial_response, initial_verdict, inst1, inst2):
    score1 = inst1['score']
    score2 = inst2['score']
    if score1 == score2:
        return None
    elif score1 > score2:
        return {
            'idx': idx,
            'question': question,
            'initial_response': initial_response,
            'initial_verdict': initial_verdict,
            'feedback_w': inst1['feedback'],
            'feedback_l': inst2['feedback'],
            'w_score': score1,
            'l_score': score2,
        }
    else:
        return {
            'idx': idx,
            'question': question,
            'initial_response': initial_response,
            'initial_verdict': initial_verdict,
            'feedback_w': inst2['feedback'],
            'feedback_l': inst1['feedback'],
            'w_score': score2,
            'l_score': score1,
        }

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial_response_path", type=str)
    parser.add_argument("--crm_data_path", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()


    initial_verdicts = {}
    # 1. save initial_response_verdict at dictionary
    with open(args.initial_response_path, "r") as f:
        verdicts = f.readlines()
    
    for verdict in verdicts:
        inst = json.loads(verdict)
        initial_verdicts[inst['idx']] = inst['correct']


    # 2. read initial_response path
    with open(args.crm_data_path, "r") as f:
        crm_lines = f.readlines()

    crms = defaultdict(list)

    for line in crm_lines:
        inst = json.loads(line)
        idx = inst['idx']
        if idx not in crms:
            crms[idx] = []
        crms[idx].append(inst)



    pairwise_data=[]
    correct_pairwise_data=[]
    wrong_pairwise_data=[]
    for key in crms:
        question, initial_response = crms[key][0]['question'], crms[key][0]['initial_response']
        initial_verdict = initial_verdicts[key]

        inst1, inst2, inst3 = crms[key][0], crms[key][1], crms[key][2]

        new_pair = cmp_func(key, question, initial_response, initial_verdict, inst1, inst2)
        if new_pair is not None:
            pairwise_data.append(new_pair)
            if initial_verdict:
                correct_pairwise_data.append(new_pair)
            else:
                wrong_pairwise_data.append(new_pair)
        # cmp 2,3
        new_pair = cmp_func(key, question, initial_response, initial_verdict, inst2, inst3)
        if new_pair is not None:
            pairwise_data.append(new_pair)
            if initial_verdict:
                correct_pairwise_data.append(new_pair)
            else:
                wrong_pairwise_data.append(new_pair)
        # cmp 1,3
        new_pair = cmp_func(key, question, initial_response, initial_verdict, inst1, inst3)
        if new_pair is not None:
            pairwise_data.append(new_pair)
            if initial_verdict:
                correct_pairwise_data.append(new_pair)
            else:
                wrong_pairwise_data.append(new_pair)
    
    
    # 3. output all files
    with open(f"{args.dataset}_pairwise_full.jsonl", "w") as f:
        for inst in pairwise_data:
            f.write(json.dumps(inst)+'\n')

    with open(f"{args.dataset}_pairwise_correct.jsonl", "w") as f:
        for inst in correct_pairwise_data:
            f.write(json.dumps(inst)+'\n')
    
    with open(f"{args.dataset}_pairwise_wrong.jsonl", "w") as f:
        for inst in wrong_pairwise_data:
            f.write(json.dumps(inst)+'\n')

    print("Done")
    
