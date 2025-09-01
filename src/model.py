import torch
import torch.nn as nn
from src.modules import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    """
    One Transformer encoder layer:
      1) Multi-head self-attention + residual + layer norm
      2) Feed-forward network + residual + layer norm
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor | None = None):
        # 1. Self-attention sublayer
        attn_out, _ = self.self_attn(x, x, attn_mask=src_mask)
        x = self.norm1(x + self.drop(attn_out))

        # 2. Feed-forward sublayer
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))

        return x
