import torch.nn as nn
from transformers import AutoModel

class SequenceModel(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels=2, dropout: float =0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, return_dict=True, output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_masks):
        out = self.encoder(input_ids, attention_masks)[1]
        out = self.dropout(out)
        out = self.head(out)
        return out