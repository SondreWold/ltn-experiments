import argparse
import random
import torch.nn as nn
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score
from datasets import Dataset as Ds
import numpy as np
import ltn
import logging 
from model import SequenceModel
from dataset import dataset_creator
import os
import sklearn.metrics as metrics

os.environ["TOKENIZERS_PARALLELISM"] = "True"
device = 'cuda:0' if torch.cuda.is_available() else 'mps'

number_of_labels = {
    'RTE': 2,
    'MNLI': 3,
    'QNLI': 2,
    'COLA': 2
    }

task_to_metric = {
    'RTE':  metrics.accuracy_score,
    'MNLI': metrics.accuracy_score,
    'QNLI': metrics.accuracy_score,
    'COLA': metrics.matthews_corrcoef
    }

def parse_args():
    parser = argparse.ArgumentParser(description="ArgumentParser for GLUE scripts")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use during training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay to use during training.") 
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="The pretrained model to use")
    parser.add_argument("--task", type=str, default="rte", help="The task to train on")
    parser.add_argument("--epochs", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--debug", action='store_true', help="Trigger debug mode")
    parser.add_argument("--freeze", action='store_true', help="Only fine tune classification head")
    parser.add_argument("--test", action='store_true', help="Trigger test eval")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate).")
    parser.add_argument("--seed", type=int, default=42, help="The rng seed")
    parser.add_argument("--gradient_clip", action='store_true', help="The gradient clip")
    parser.add_argument("--beta", type=float, default=1, help="The adam momentum")
    parser.add_argument("--patience", type=int, default=2, help="The patience value")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout value")

    args = parser.parse_args()
    return args

def main(args):
    print("====================================================================================================")
    logging.info(f"Initialised training on task: {args.task.upper()}, debug={args.debug}, device={device}")
    print("====================================================================================================")
    print("\n")

    config = {
            "task": args.task,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "model": args.model_name,
            "frozen": args.freeze,
            "test": args.test,
            "dropout": args.dropout,
        }
    
    
    train_dataset = dataset_creator(args.task)(tokenizer_path_or_name=args.model_name, root_data_path="./data", split='train', max_length=128)
    val_dataset = dataset_creator(args.task)(tokenizer_path_or_name=args.model_name, root_data_path="./data", split='dev', max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    logging.info(train_dataset.get_decoded_example(13))

    model = SequenceModel(args.model_name, number_of_labels[args.task.upper()], args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    for epoch in range(10):
        logging.info(f"Started training at epoch {epoch}")
        model.train()
        train_loss = 0.
        for i, (input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
            out = model(input_ids, attention_masks)
            loss = criterion(out, y.squeeze(-1))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.
            golds = []
            preds = []
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
                out = model(input_ids, attention_masks)
                loss = criterion(out, y.squeeze(-1))
                val_loss += loss.item()
                golds.extend(y.cpu().tolist())
                y_hat = torch.argmax(out.cpu(), dim=-1)
                preds.extend(y_hat.tolist())
            
            score = task_to_metric[args.task.upper()](golds, preds)
      
        logging.info(f"LOSS - train: {(train_loss/len(train_loader)):.3f}, valid: {(val_loss/len(val_loader)):.3f}, METRIC SCORE - val: {score} ")

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main(args)