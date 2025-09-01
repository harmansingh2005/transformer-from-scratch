
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

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of parallel attention heads.
        dropout (float): dropout on attention weights.

    Input shape:
        x_q, x_kv: [B, T, d_model]
    Output shape:
        out: [B, T, d_model]
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for queries, keys, values
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Output projection
        self.Wo = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_k, dropout)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split last dim (d_model) into (num_heads, d_k).
        x: [B, T, d_model] -> [B, h, T, d_k]
        """
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge heads back: [B, h, T, d_k] -> [B, T, d_model]
        """
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, x_q: torch.Tensor, x_kv: torch.Tensor,
                attn_mask: torch.Tensor | None = None):
        # Project to Q,K,V
        Q = self._split_heads(self.Wq(x_q))  # [B,h,Tq,d_k]
        K = self._split_heads(self.Wk(x_kv)) # [B,h,Tk,d_k]
        V = self._split_heads(self.Wv(x_kv)) # [B,h,Tk,d_k]

        # Apply scaled dot-product attention
        out, attn = self.attn(Q, K, V, attn_mask)  # out: [B,h,Tq,d_k]

        # Merge heads and project
        out = self._merge_heads(out)   # [B,Tq,d_model]
        out = self.Wo(out)             # final linear projection
        return self.drop(out), attn

