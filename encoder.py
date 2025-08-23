from turtle import forward
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_heads, dropout)

        self.self_attention_norm = nn.LayerNorm(d_model)

        self.feed_forward_network = FeedForward(d_model, d_ff, dropout)

        self.feed_forward_network_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        x = self.self_attention(x, x, x, src_mask)
        x = self.self_attention_norm(x)

        x = self.feed_forward_network(x)
        x = self.feed_forward_network_norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_encoders) -> None:
        super().__init__()

        self.encoding_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads) for _ in range(num_encoders)]
        )

    def forward(self, src, src_mask):
        output = src
        for layer in self.encoding_layers:
            output = layer(output, src_mask)
        return output
