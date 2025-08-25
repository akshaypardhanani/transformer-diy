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

        self.last_attn = None

    def forward(self, query, key, value, mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        print("DEBUG SA: q", q.shape, "k", k.shape, "v", v.shape)

        dim_k = k.size(-1)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(dim_k)

        print("DEBUG SA: scores", scores.shape)

        if mask is not None:
            print("DEBUG SA: mask raw", mask.shape, mask.dtype)
            scores = scores.masked_fill(mask == 0, -float("inf"))

        weights = F.softmax(scores, dim=-1)
        print("DEBUG SA: weights", weights.shape)
        weights = self.dropout(weights)
        print("DEBUG SA: weights dropout", weights.shape)

        outputs = torch.bmm(weights, v)
        print("DEBUG SA: outputs", outputs.shape)
        return outputs
