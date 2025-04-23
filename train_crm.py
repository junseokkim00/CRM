import argparse
import torch
from torch.optim import Adam
import torch.nn.functional as F
from crm_utils import get_dataloader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import json



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_file", type=str)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    model = model.to(device)

    train_jsonl_path="./logs/gsm8k_crm_train.jsonl"
    test_jsonl_path="./logs/gsm8k_crm_test.jsonl"

    print("get dataloader")
    train_dataloader = get_dataloader(train_jsonl_path, tokenizer, device, batch_size=args.batch_size)
    test_dataloader = get_dataloader(test_jsonl_path, tokenizer, device, batch_size=args.batch_size)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # train the model with data

    best_mse_loss = -1

    for epoch in range(args.epoch):
        model.train()
        average_train_loss = 0
        for batch in tqdm(train_dataloader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] =batch['labels'].to(device)

            outputs = model(**batch)
            loss = outputs.loss
            average_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del outputs, loss

        average_train_loss /= len(train_dataloader)
        print(f"Epoch {epoch} - Average Training Loss: {average_train_loss:.4f}")

        # evaluate
        model.eval()
        preds, golds = [], []
        for batch in tqdm(test_dataloader):
            batch['input_ids'] = batch['input_ids'].to(device)
            batch['attention_mask'] = batch['attention_mask'].to(device)
            batch['labels'] =batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(**batch)
                predictions = outputs.logits
                preds.extend(predictions.cpu().numpy())
                golds.extend(batch['labels'].cpu().numpy())
        
        mse = F.mse_loss(torch.tensor(preds), torch.tensor(golds)).item()
        print(f"Epoch {epoch} - Test MSE loss: {mse}")


        with open("./gsm8k_crm_deberta_v3_lr:{args.lr}_large.jsonl", "a+") as f:
            inst = {
                'epoch': epoch,
                'train_loss': average_train_loss,
                'test_mse': mse,
                'best_test_mse': best_mse_loss
            }
            f.write(json.dumps(inst) + '\n')

        if best_mse_loss == -1 or best_mse_loss >= mse:
            best_mse_loss = mse
            model.save_pretrained(f"./{args.save_file}")