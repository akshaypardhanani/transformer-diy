import math
import torch


from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, output_size, dropout=0.1) -> None:
        super().__init__()

        self.query = nn.Linear(d_model, output_size)
        self.key = nn.Linear(d_model, output_size)
        self.value = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        dim_k = k.size(-1)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim_k)

        if mask:
            scores = scores.masked_fill(mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        outputs = torch.bmm(weights, v)
        return outputs
