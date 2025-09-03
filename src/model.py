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
        x = self.embed(src_tokens) * (self.embed.embedding_dim ** 0.5)       # [B, T, d_model]
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

class Decoder(nn.Module):
    """
    Full Transformer Decoder:
      - Token embedding + positional encoding
      - Stack of N DecoderLayer blocks (masked self-attn + cross-attn + FFN)
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
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def make_self_mask(self, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Combine look-ahead mask with target padding mask. Returns [B,1,T,T]."""
        T = tgt_tokens.size(1)
        look_ahead = DecoderLayer.make_look_ahead_mask(T, device=tgt_tokens.device)   # [1,1,T,T]
        pad_mask  = DecoderLayer.make_tgt_padding_mask(tgt_tokens, self.pad_id)       # [B,1,T,T]
        return look_ahead | pad_mask

    @staticmethod
    def make_cross_mask(tgt_tokens: torch.Tensor, src_tokens: torch.Tensor, pad_id_src: int) -> torch.Tensor:
        """Mask encoder padding positions for cross-attention. Returns [B,1,T_dec,T_enc]."""
        return DecoderLayer.make_cross_padding_mask(tgt_tokens, src_tokens, pad_id_src)

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor,
        cross_attn_mask: torch.Tensor
    ) -> torch.Tensor:
        y = self.embed(tgt_tokens) * (self.embed.embedding_dim ** 0.5)      # [B,T,D]
        y = self.pos(y)
        for layer in self.layers:
            y = layer(y, memory, self_attn_mask, cross_attn_mask)
        return self.norm(y)


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer with generator head.

    forward(src_tokens, tgt_tokens) returns logits over target vocab for next-token prediction:
      - We feed decoder with tgt[:, :-1] (teacher forcing)
      - We output logits for positions predicting tgt[:, 1:]
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers_enc: int = 6,
        num_layers_dec: int = 6,  
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        pad_id_src: int = 0,
        pad_id_tgt: int = 0,
    ):
        super().__init__()
        self.pad_id_src = pad_id_src
        self.pad_id_tgt = pad_id_tgt

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_layers=num_layers_enc,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pad_id=pad_id_src,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers_dec,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pad_id=pad_id_tgt,
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)  # Final linear layer
        self.generator.weight = self.decoder.embed.weight  # weight tying

    def encode(self, src_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (memory, src_mask)."""
        src_mask = self.encoder.make_src_padding_mask(src_tokens, self.pad_id_src)  # [B,1,S,S]
        memory = self.encoder(src_tokens, src_mask)  # [B,S,D]
        return memory, src_mask

    def decode(self, tgt_tokens_in: torch.Tensor, memory: torch.Tensor, src_tokens: torch.Tensor) -> torch.Tensor:
        """Run decoder and return hidden states [B,T_dec,D]."""
        self_mask  = self.decoder.make_self_mask(tgt_tokens_in)  # [B,1,T,T]
        cross_mask = self.decoder.make_cross_mask(tgt_tokens_in, src_tokens, self.pad_id_src)  # [B,1,T_dec,T_enc]
        return self.decoder(tgt_tokens_in, memory, self_mask, cross_mask)

    def forward(self, src_tokens: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_tokens: [B, S]
            tgt_tokens: [B, T] including BOS at position 0 (and EOS/pad later)

        Returns:
            logits: [B, T-1, vocab_tgt] (for predicting tokens 1..T-1)
        """
        memory, _ = self.encode(src_tokens)
        # Teacher forcing: input to decoder excludes the last token
        tgt_in = tgt_tokens[:, :-1]            # [B, T-1]
        dec_h = self.decode(tgt_in, memory, src_tokens)  # [B, T-1, D]
        logits = self.generator(dec_h)         # [B, T-1, Vtgt]
        return logits
