from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch
from typing import Tuple, List

class RTE_Dataset(Dataset):

    def __init__(self, tokenizer_path_or_name: str = 'bert-base-uncased', root_data_path: str = "./data", split: str = 'train', max_length: int=128):
        '''
            Loads and processes a RTE dataset
        '''
        assert split in ['dev', 'train', 'test']
        if root_data_path[-1] == '/':
            root_data_path = root_data_path[:-1]

        label_converter  = {"entailment": 0, "not_entailment": 1}
        self.label_inverter = {v:k for k,v in label_converter.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        DATASET_PATH = f'{root_data_path}/RTE/{split}.tsv'

        df = pd.read_csv(DATASET_PATH, sep='\t', header=0)
        df = df[df['label'].notna()]
        self.labels = [label_converter[l] for l in df['label'].to_list()]

        self.features = self.tokenizer(
            df['sentence1'].to_list(),
            df['sentence2'].to_list(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            )

        del df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), torch.tensor([self.labels[index]])

    def get_decoded_example(self, index) -> str:
        return f'EXAMPLE: {self.tokenizer.decode(self.features["input_ids"][index])}, LABEL: {self.labels[index]} = {self.label_inverter[self.labels[index]]}'

class MNLI_Dataset(Dataset):

    def __init__(self, tokenizer_path_or_name: str = 'bert-base-uncased', root_data_path: str = "./data", split: str = 'train', max_length: int=128):
        '''
            Loads and processes the MNLI dataset
        '''
        assert split in ['dev', 'train', 'test']
        if root_data_path[-1] == '/':
            root_data_path = root_data_path[:-1]

        label_converter  = {"entailment": 0, "contradiction": 1, "neutral": 2}
        self.label_inverter = {v:k for k,v in label_converter.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        DATASET_PATH = f'{root_data_path}/MNLI/{split}.tsv' if split == 'train' else f'{root_data_path}/MNLI/{split}_matched.tsv'

        df = pd.read_csv(DATASET_PATH, sep='\t', header=0, on_bad_lines='skip')
        df = df[df['gold_label'].notna()]
        df = df[df['sentence1'].notna()]
        df = df[df['sentence2'].notna()]

        self.labels = [label_converter[l] for l in df['gold_label'].to_list()]

        self.features = self.tokenizer(
            df['sentence1'].to_list(),
            df['sentence2'].to_list(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            )

        del df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), torch.tensor([self.labels[index]])
    
    def get_decoded_example(self, index) -> str:
        return f'EXAMPLE: {self.tokenizer.decode(self.features["input_ids"][index])}, LABEL: {self.labels[index]} = {self.label_inverter[self.labels[index]]}'

class QNLI_Dataset(Dataset):

    def __init__(self, tokenizer_path_or_name: str = 'bert-base-uncased', root_data_path: str = "./data", split: str = 'train', max_length: int=128):
        '''
            Loads and processes the MNLI dataset
        '''
        assert split in ['dev', 'train', 'test']
        if root_data_path[-1] == '/':
            root_data_path = root_data_path[:-1]

        label_converter  = {"entailment": 0, "not_entailment": 1}
        self.label_inverter = {v:k for k,v in label_converter.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        DATASET_PATH = f'{root_data_path}/QNLI/{split}.tsv'

        df = pd.read_csv(DATASET_PATH, sep='\t', header=0, on_bad_lines='skip')
        df = df[df['label'].notna()]
        df = df[df['question'].notna()]
        df = df[df['sentence'].notna()]

        self.labels = [label_converter[l] for l in df['label'].to_list()]

        self.features = self.tokenizer(
            df['question'].to_list(),
            df['sentence'].to_list(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            )

        del df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), torch.tensor([self.labels[index]])
    
    def get_decoded_example(self, index) -> str:
        return f'EXAMPLE: {self.tokenizer.decode(self.features["input_ids"][index])}, LABEL: {self.labels[index]} = {self.label_inverter[self.labels[index]]}'

class COLA_Dataset(Dataset):

    def __init__(self, tokenizer_path_or_name: str = 'bert-base-uncased', root_data_path: str = "./data", split: str = 'train', max_length: int=128):
        '''
            Loads and processes the MNLI dataset
        '''
        assert split in ['dev', 'train', 'test']
        if root_data_path[-1] == '/':
            root_data_path = root_data_path[:-1]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        DATASET_PATH = f'{root_data_path}/CoLA/{split}.tsv'

        df = pd.read_csv(DATASET_PATH, sep='\t', header=0 ,on_bad_lines='skip', names=["source", "label", "skip", "sentence"])
        df = df[df['label'].notna()]
        df = df[df['sentence'].notna()]

        self.labels =  df['label'].to_list()

        self.features = self.tokenizer(
            df['sentence'].to_list(),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            )

        del df

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.LongTensor, torch.BoolTensor, torch.Tensor]:
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), torch.tensor([self.labels[index]])
    
    def get_decoded_example(self, index) -> str:
        return f'EXAMPLE: {self.tokenizer.decode(self.features["input_ids"][index])}, LABEL: {self.labels[index]} = {self.labels[index]}'

def dataset_creator(task:str) -> Dataset:
    options = {'RTE': RTE_Dataset, 'MNLI': MNLI_Dataset, 'QNLI': QNLI_Dataset, 'COLA': COLA_Dataset}
    assert task.upper() in options.keys()
    return options[task.upper()]

if __name__ == '__main__':
    print("Loaded!")

    #rte = RTE_Dataset()
    #i,a,l = rte[0:16]
    #dec = rte.tokenizer.decode(i)
    #print(dec, rte.label_inverter[l])