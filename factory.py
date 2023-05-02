import ltn
from typing import List
import torch

def get_constants(task: str, device: str) -> List[ltn.Constant]:
    tasks = [
    'RTE',
    'QNLI',
    'COLA',
    'SST',
    'WNLI'
    ]
    if task.upper() in ['MNLI']:
        return [
            ltn.Constant(torch.tensor([1, 0, 0]).to(device)),
            ltn.Constant(torch.tensor([0, 1, 0]).to(device)),
            ltn.Constant(torch.tensor([0, 0, 1]).to(device))
        ]
    else:
        return [
            ltn.Constant(torch.tensor([1, 0]).to(device)),
            ltn.Constant(torch.tensor([0, 1]).to(device))
        ]
