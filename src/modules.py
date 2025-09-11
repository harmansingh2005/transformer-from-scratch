import math
import torch  # type: ignore
import torch.nn as nn # type: ignore


def positional_encoding(x, max_len= 10000):
    """
    Adds positional encodings directly to input tensor `x`.
    x shape: [B, T, d_model]
    """
    B, T, d_model = x.size()

    pe = torch.zeros(max_len, d_model, device=x.device, dtype=x.dtype)
    position = torch.arange(0, max_len, dtype=torch.float32, device=x.device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=x.device) * 
                         (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pos = pe[:T, :].unsqueeze(0) 
    return x + pos

class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_k, dropout = 0.1):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_k)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask: torch.Tensor | None = None):
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # Attention scores [B,h,Tq,Tk]
        if attn_mask is not None:
            neg_large = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attn_mask, neg_large)
        attn = torch.softmax(scores, dim=-1) # Normalization 
        attn = self.drop(attn)
        output = torch.matmul(attn, V) # Attention final output
        return output

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention layer*.

    Args:
        d_model (int): embedding dimension.
        num_heads (int): number of parallel attention heads.
        dropout (float): dropout on attention weights.

    Input shape:
        x_q, x_kv: [B, T, d_model]
    Output shape:
        out: [B, T, d_model]
    """

    def __init__(self, d_model, num_heads, dropout = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads

        # Linear layers
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        # Final linear layer
        self.Wo = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(self.d_k, dropout)
        self.drop = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x):
        B, h, T, d_k = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d_k)

    def forward(self, x_q, x_kv, attn_mask: torch.Tensor | None = None):
        Q = self._split_heads(self.Wq(x_q))  # [B,h,Tq,d_k]
        K = self._split_heads(self.Wk(x_kv)) # [B,h,Tk,d_k]
        V = self._split_heads(self.Wv(x_kv)) # [B,h,Tk,d_k]

        out, attn = self.attn(Q, K, V, attn_mask)  # out: [B,h,Tq,d_k]

        out = self._merge_heads(out)  
        out = self.Wo(out)            
        return self.drop(out), attn


class PositionwiseFeedForward(nn.Module):
    # FeedForward Neural Network
    def __init__(self, d_model, d_ff = 2048, dropout = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.drop(self.act(self.linear1(x))))


