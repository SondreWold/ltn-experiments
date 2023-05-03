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
from torch.optim import lr_scheduler
from model import SequenceModel, LogitsToPredicate
from dataset import dataset_creator
import os
import sklearn.metrics as metrics
import pprint
import ltn
import wandb
import math
from factory import get_constants

os.environ["TOKENIZERS_PARALLELISM"] = "True"
device = 'cuda:0' if torch.cuda.is_available() else 'mps'

number_of_labels = {
    'RTE': 2,
    'MNLI': 3,
    'QNLI': 2,
    'COLA': 2,
    'SST': 2,
    'WNLI': 2
    }

task_to_metric = {
    'RTE':  metrics.accuracy_score,
    'MNLI': metrics.accuracy_score,
    'QNLI': metrics.accuracy_score,
    'COLA': metrics.matthews_corrcoef,
    'SST': metrics.accuracy_score,
    'WNLI': metrics.accuracy_score
    }

def parse_args():
    parser = argparse.ArgumentParser(description="ArgumentParser for GLUE scripts")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use during training.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="The weight decay to use during training.") 
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="The pretrained model to use")
    parser.add_argument("--task", type=str, default="rte", help="The task to train on")
    parser.add_argument("--optimizer", type=str, default="adamw", help="The optimizer function to train with")
    parser.add_argument("--scheduler", type=str, default="cosine", help="The scheduler to train with")
    parser.add_argument("--epochs", type=int, default=3, help="The number of epochs.")
    parser.add_argument("--p_value", type=int, default=2, help="The p value for pMeanAgg")
    parser.add_argument("--debug", action='store_true', help="Trigger debug mode")
    parser.add_argument("--logic_mode", action='store_true', help="Trigger the use of LTN and SatAgg objective")
    parser.add_argument("--freeze", action='store_true', help="Only fine tune classification head")
    parser.add_argument("--test", action='store_true', help="Trigger test eval")
    parser.add_argument("--lr", type=float, default=5e-5, help="The learning rate).")
    parser.add_argument("--seed", type=int, default=42, help="The rng seed")
    parser.add_argument("--gradient_clip", action='store_true', help="The gradient clip")
    parser.add_argument("--beta", type=float, default=1, help="The adam momentum")
    parser.add_argument("--patience", type=int, default=2, help="The patience value")
    parser.add_argument("--dropout", type=float, default=0.2, help="The dropout value")
    parser.add_argument("--sample", type=float, default=0.2, help="The percentage of the original dataset to use for training")
    parser.add_argument("--max_steps", default=31250, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion", default=0.01, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

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
            "scheduler": args.scheduler,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "model": args.model_name,
            "freeze": args.freeze,
            "test": args.test,
            "dropout": args.dropout,
            "logic_mode": args.logic_mode,
            "p_value": args.p_value
        }
    
    if not args.debug:
        wandb.init(project="ltn", config=config, entity="sondrewo")

    logging.info("HYPERPARAMETERS: \n")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print("\n")
    
    train_dataset = dataset_creator(args.task)(tokenizer_path_or_name=args.model_name, root_data_path="./data", split='train', max_length=128)
    val_dataset = dataset_creator(args.task)(tokenizer_path_or_name=args.model_name, root_data_path="./data", split='dev', max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    logging.info(train_dataset.get_decoded_example(13))

    model = SequenceModel(device, args.model_name, number_of_labels[args.task.upper()], args.dropout).to(device)

    if args.logic_mode:
        P = ltn.Predicate(LogitsToPredicate(device, model)).to(device)
        Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=args.p_value), quantifier="f")
        SatAgg = ltn.fuzzy_ops.SatAgg()
        constants = get_constants(args.task, device)

    params = list(model.named_parameters())
    no_decay = ['bias', 'layer_norm', 'embedding']
    decay_params = [(n, p) for n, p in params if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in params if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
    else:
        pass

    max_steps = len(train_loader) * args.epochs

    if args.scheduler == "cosine":
        def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

                return lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = cosine_schedule_with_warmup(optimizer, int(max_steps * args.warmup_proportion), max_steps, 0.1)

    elif args.scheduler == "linear":
        scheduler = lr_scheduler.ChainedScheduler([
            lr_scheduler.LinearLR(optimizer, start_factor=1e-9, end_factor=1.0, total_iters=int(max_steps * args.warmup_proportion)),
            lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=1e-9, total_iters=max_steps)
        ])

    criterion = CrossEntropyLoss()

    if args.freeze:
        for param in model.encoder.parameters():
            param.requires_grad = False

    for epoch in range(args.epochs):
        logging.info(f"Started training at epoch {epoch}")
        model.train()
        train_loss = 0.
        running_losses = []
        mean_sat = 0.
        for i, (input_ids, attention_masks, y) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
            y = y.squeeze(-1)
            data = torch.cat((input_ids, attention_masks), -1).to(device)
            if args.logic_mode:
                if number_of_labels[args.task.upper()] == 3:
                    x_A = ltn.Variable("x_A", data[y == 0]) # class A examples
                    x_B = ltn.Variable("x_B", data[y == 1]) # class B examples
                    x_C = ltn.Variable("x_B", data[y == 2]) # class C examples
                    satt_agg = SatAgg(
                        Forall(x_A, P(x_A, constants[0])),
                        Forall(x_B, P(x_B, constants[1])),
                        Forall(x_B, P(x_B, constants[2]))
                    )
                elif number_of_labels[args.task.upper()] == 2:
                    x_A = ltn.Variable("x_A", data[y == 0]) # class A examples
                    x_B = ltn.Variable("x_B", data[y == 1]) # class B examples
                    satt_agg = SatAgg(
                        Forall(x_A, P(x_A, constants[0])),
                        Forall(x_B, P(x_B, constants[1])),
                    )
                mean_sat += satt_agg
                loss = 1. - satt_agg
                train_loss += loss.item()
            else:
                out = model(data)
                loss = criterion(out, y)
            c = loss.item()
            train_loss += c
            running_losses.append(c)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if args.debug:
                break
            else:
                wandb.log({"running_loss": np.mean(running_losses)})

        model.eval()
        with torch.no_grad():
            val_loss = 0.
            mean_sat = 0.
            golds = []
            preds = []
            for i, (input_ids, attention_masks, y) in enumerate(tqdm(val_loader)):
                input_ids, attention_masks, y = input_ids.to(device), attention_masks.to(device), y.to(device)
                y = y.squeeze(-1)
                data = torch.cat((input_ids, attention_masks), -1).to(device)
                if args.logic_mode:
                    if number_of_labels[args.task.upper()] == 3:
                        x_A = ltn.Variable("x_A", data[y == 0]) # class A examples
                        x_B = ltn.Variable("x_B", data[y == 1]) # class B examples
                        x_C = ltn.Variable("x_C", data[y == 2]) # class C examples
                        satt_agg = SatAgg(
                            Forall(x_A, P(x_A, constants[0])),
                            Forall(x_B, P(x_B, constants[1])),
                            Forall(x_B, P(x_C, constants[2]))
                        )

                    elif number_of_labels[args.task.upper()] == 2:
                        x_A = ltn.Variable("x_A", data[y == 0]) # class A examples
                        x_B = ltn.Variable("x_B", data[y == 1]) # class B examples
                        satt_agg = SatAgg(
                            Forall(x_A, P(x_A, constants[0])),
                            Forall(x_B, P(x_B, constants[1])),
                        )
                    mean_sat += satt_agg
                    loss = 1. - satt_agg
                out = model(data)
                if not args.logic_mode:
                    loss = criterion(out, y)
                val_loss += loss.item()
                golds.extend(y.cpu().tolist())
                y_hat = torch.argmax(out.cpu(), dim=-1)
                preds.extend(y_hat.tolist())
                break
            
            score = task_to_metric[args.task.upper()](golds, preds)
        
        t_l = train_loss / len(train_loader)
        v_l = val_loss / len(val_loader)
      
        if not args.debug:
            wandb.log({"train_loss_epoch": t_l})
            wandb.log({"val_loss_epoch": v_l})
            if args.logic_mode:
                wandb.log({"sat_level_epoch": mean_sat / len(val_loader)})

        if args.logic_mode:
            logging.info(f"LOSS - train: {(t_l):.3f}, valid: {(v_l):.3f}, SAT - val: {mean_sat / len(val_loader):.3f}, SCORE - val: {score:.3f} ")
        else:
            logging.info(f"LOSS - train: {(t_l):.3f}, valid: {(v_l):.3f}, METRIC SCORE - val: {score:.3f} ")

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