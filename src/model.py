import torch
import torch.nn as nn
from src.modules import MultiHeadAttention, PositionwiseFeedForward
from src.modules import PositionalEncoding  

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
class Encoder(nn.Module):
    """
    Full Transformer Encoder:
      - Token embedding + positional encoding
      - Stack of N EncoderLayer blocks

    Args:
        vocab_size (int): size of source vocabulary
        d_model (int): embedding size
        num_layers (int): number of encoder layers (N)
        num_heads (int): attention heads per layer
        d_ff (int): FFN hidden size (usually 4 * d_model)
        dropout (float): dropout prob
        pad_id (int): padding token id for mask helper
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def make_src_padding_mask(src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Create padding mask for self-attention:
          True = mask (ignore), False = keep.
        Shape returned: [B, 1, T, T] to broadcast across heads.

        If you prefer [B, T, T], you can adapt in attention; we use [B,1,T,T] here.
        """
        # [B, T]
        pad_positions = (src_tokens == pad_id)
        # Expand to pairwise mask [B, T_q, T_k]
        mask_2d = pad_positions.unsqueeze(1).expand(-1, src_tokens.size(1), -1)  # [B, T, T]
        # Add head axis as 1 for broadcasting: [B, 1, T, T]
        return mask_2d.unsqueeze(1)

    def forward(self, src_tokens: torch.Tensor, src_mask: torch.Tensor | None = None):
        """
        Args:
            src_tokens: [B, T] integer token ids
            src_mask: optional [B, 1, T, T] boolean mask (True = mask)

        Returns:
            memory: [B, T, d_model] encoded representations
        """
        x = self.embed(src_tokens)        # [B, T, d_model]
        x = self.pos(x)                   # add positional encodings
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """
    One Transformer decoder layer:
      1) Masked multi-head self-attention  (prevent looking ahead)
      2) Cross-attention over encoder outputs
      3) Position-wise feed-forward
      Each sublayer uses residual + LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def make_look_ahead_mask(T: int, device=None) -> torch.Tensor:
        """
        Upper-triangular True mask (excluding diagonal) to block future positions.
        Shape: [1, 1, T, T] so it can broadcast to [B, h, T, T].
        """
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    @staticmethod
    def make_tgt_padding_mask(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Boolean mask (True = mask) for target padding positions, expanded to [B,1,T,T].
        """
        pad = (tgt_tokens == pad_id)  # [B,T]
        return pad.unsqueeze(1).expand(-1, tgt_tokens.size(1), -1).unsqueeze(1)  # [B,1,T,T]

    @staticmethod
    def make_cross_padding_mask(tgt_tokens: torch.Tensor, src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        Cross-attention mask: mask encoder positions that are padding.
        Returns [B,1,T_dec,T_enc].
        """
        src_pad = (src_tokens == pad_id)  # [B,S]
        return src_pad.unsqueeze(1).expand(-1, tgt_tokens.size(1), -1).unsqueeze(1)  # [B,1,T,S]

    def forward(
        self,
        y: torch.Tensor,                 # [B, T_dec, d_model]
        memory: torch.Tensor,            # [B, T_enc, d_model]
        self_attn_mask: torch.Tensor,    # [B,1,T_dec,T_dec] (True = mask)
        cross_attn_mask: torch.Tensor    # [B,1,T_dec,T_enc] (True = mask)
    ) -> torch.Tensor:
        # 1) Masked self-attn
        self_out, _ = self.self_attn(y, y, self_attn_mask)
        y = self.norm1(y + self.drop(self_out))

        # 2) Cross-attn over encoder memory
        cross_out, _ = self.cross_attn(y, memory, cross_attn_mask)
        y = self.norm2(y + self.drop(cross_out))

        # 3) FFN
        ffn_out = self.ffn(y)
        y = self.norm3(y + self.drop(ffn_out))

        return y
