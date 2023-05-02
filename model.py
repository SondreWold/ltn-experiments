import torch.nn as nn
from transformers import AutoModel
import torch

class SequenceModel(nn.Module):
    def __init__(self, device, model_name: str="bert-base-uncased", num_labels=2, dropout: float=0.1):
        super().__init__()
        self.device = device
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, data):
        l = data.size()[-1] // 2
        input_ids = data[:, :l].to(self.device)
        attention_masks = data[:, l:].to(self.device)
        out = self.encoder(input_ids, attention_masks)[1]
        out = self.dropout(out)
        out = self.head(out)
        return out


class LogitsToPredicate(nn.Module):
    def __init__(self, device, logits_model):
        super(LogitsToPredicate, self).__init__()
        self.device = device
        self.logits_model = logits_model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, l, training=False):
        l = l.to(self.device)
        logits = self.logits_model(x)
        probs = self.softmax(logits)
        out = torch.sum(probs * l, dim=1)
        return out