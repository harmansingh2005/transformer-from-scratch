
import math
import torch 
import torch.nn as nn # type: ignore


class PositionalEncoding(nn.Module):
    """
    Args:
        d_model (int): embedding dimension.
        dropout (float): dropout probability applied after adding PE.
        max_len (int): maximum sequence length supported.

    Shape:
        - Input:  x of shape [batch, seq_len, d_model]
        - Output: same shape as input

    Notes:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        Stored as a buffer so it’s on the right device and in checkpoints,
        but not a trainable parameter.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)         # for even indices
        pe[:, 1::2] = torch.cos(position * div_term)         # for odd indices

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pos = self.pe[:seq_len, :].unsqueeze(0).to(dtype=x.dtype, device=x.device)
        x = x + pos # adding position encoding into each token’s embedding.
        return self.dropout(x)
class ScaledDotProductAttention(nn.Module):
    """
    $ Core attention primitive $

    Given Q, K, V:
      scores = Q @ K^T / sqrt(d_k)
      attn   = softmax(scores, dim=-1)
      out    = attn @ V

    Args:
      d_k (int): key/query depth for scaling.
      dropout (float): dropout on attention weights.

    Shapes:
      Q, K, V: [B, h, Tq, d_k], [B, h, Tk, d_k], [B, h, Tk, d_k]
      attn_mask: [B, 1, Tq, Tk] or [B, h, Tq, Tk]
      Output: out [B, h, Tq, d_k], attn [B, h, Tq, Tk]
    """
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                attn_mask: torch.Tensor | None = None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # Attention scores [B,h,Tq,Tk]
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)                         # Normalization(softmax)
        attn = self.drop(attn)
        out = torch.matmul(attn, V)                                  # final vector [B, h, Tq, d_k]
        return out, attn

