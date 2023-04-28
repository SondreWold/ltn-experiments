from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import torch

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

    def __getitem__(self, index):
        return torch.LongTensor(self.features["input_ids"][index]), torch.BoolTensor(self.features["attention_mask"][index]), torch.tensor([self.labels[index]])

def dataset_creator(task:str) -> Dataset:
    options = {"RTE": RTE_Dataset}
    assert task.upper() in options.keys()
    return options[task.upper()]

if __name__ == '__main__':
    print("Loaded!")

    #rte = RTE_Dataset()
    #i,a,l = rte[0:16]
    #dec = rte.tokenizer.decode(i)
    #print(dec, rte.label_inverter[l])