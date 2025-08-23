import torch


from torch import nn


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_output_size = self.d_models // self.num_heads

        self.attentions = nn.ModuleList(
            [
                SelfAttention(d_model, self.attention_output_size)
                for _ in range(self.num_heads)
            ]
        )

        self.output = nn.Linear(self.d_model, self.d_model)

    def forward(self, query, key, value, mask=None):
        x = torch.cat(
            [layer(query, key, value, mask) for layer in self.attentions], dim=-1
        )
        x = self.output(x)
        return x
