import math
import torch # type: ignore
import torch.nn as nn # type: ignore
from src.modules import MultiHeadAttention, PositionwiseFeedForward, positional_encoding


# ---------------- Encoder ----------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)
        
        self.self_attn_layers = nn.ModuleList([MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.ffn_layers       = nn.ModuleList([PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_layers)])
        self.norm1_layers     = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2_layers     = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop_layers      = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    @staticmethod
    def make_src_padding_mask(src_tokens, pad_id):
        # True = mask. Shape [B,1,T,T]
        pad = (src_tokens == pad_id)                 
        return pad.unsqueeze(1).expand(-1, src_tokens.size(1), -1).unsqueeze(1)

    def forward(self, src_tokens, src_mask=None):
        x = self.embed(src_tokens) * math.sqrt(self.embed.embedding_dim)    
        x = positional_encoding(x)
        x = self.drop(x)

        for i in range(self.num_layers):
            attn_out, _ = self.self_attn_layers[i](x, x, attn_mask=src_mask) 
            x = self.norm1_layers[i](x + self.drop_layers[i](attn_out))

            ffn_out = self.ffn_layers[i](x)                                   
            x = self.norm2_layers[i](x + self.drop_layers[i](ffn_out))

        return self.norm_final(x)


# ---------------- Decoder ----------------
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_final = nn.LayerNorm(d_model)

        # stacks (no separate DecoderLayer class)
        self.self_attn_layers  = nn.ModuleList([MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.cross_attn_layers = nn.ModuleList([MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)])
        self.ffn_layers        = nn.ModuleList([PositionwiseFeedForward(d_model, d_ff, dropout) for _ in range(num_layers)])
        self.norm1_layers      = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm2_layers      = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm3_layers      = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.drop_layers       = nn.ModuleList([nn.Dropout(dropout) for _ in range(num_layers)])

    @staticmethod
    def make_look_ahead_mask(T, device=None):
        # [1,1,T,T] True above diagonal (block future)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        return m.unsqueeze(0).unsqueeze(0)

    def make_tgt_padding_mask(self, tgt_tokens):
        pad = (tgt_tokens == self.pad_id)             # [B,T]
        return pad.unsqueeze(1).expand(-1, tgt_tokens.size(1), -1).unsqueeze(1)  # [B,1,T,T]

    @staticmethod
    def make_cross_padding_mask(tgt_tokens, src_tokens, pad_id_src):
        # [B,1,T_dec,T_enc] True where src is PAD
        src_pad = (src_tokens == pad_id_src)  # [B,S]
        return src_pad.unsqueeze(1).expand(-1, tgt_tokens.size(1), -1).unsqueeze(1)

    def make_self_mask(self, tgt_tokens):
        T = tgt_tokens.size(1)
        la  = self.make_look_ahead_mask(T, device=tgt_tokens.device)   # [1,1,T,T]
        pad = self.make_tgt_padding_mask(tgt_tokens)                   # [B,1,T,T]
        return la | pad

    def forward(self, tgt_tokens, memory, self_attn_mask, cross_attn_mask):
        y = self.embed(tgt_tokens) * math.sqrt(self.embed.embedding_dim)  # [B,T,D]
        y = positional_encoding(y)
        y = self.drop(y)

        for i in range(self.num_layers):
            self_out, _ = self.self_attn_layers[i](y, y, attn_mask=self_attn_mask)      # [B,T,D]
            y = self.norm1_layers[i](y + self.drop_layers[i](self_out))

            cross_out, _ = self.cross_attn_layers[i](y, memory, attn_mask=cross_attn_mask)
            y = self.norm2_layers[i](y + self.drop_layers[i](cross_out))

            ffn_out = self.ffn_layers[i](y)
            y = self.norm3_layers[i](y + self.drop_layers[i](ffn_out))

        return self.norm_final(y)


# ---------------- Full Transformer ----------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_layers_enc=6,
        num_layers_dec=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        pad_id_src=0,
        pad_id_tgt=0,
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

        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # weight tying with decoder embedding
        self.generator.weight = self.decoder.embed.weight

    def encode(self, src_tokens):
        src_mask = self.encoder.make_src_padding_mask(src_tokens, self.pad_id_src)  # [B,1,S,S]
        memory = self.encoder(src_tokens, src_mask)                                  # [B,S,D]
        return memory, src_mask

    def decode(self, tgt_tokens_in, memory, src_tokens):
        self_mask  = self.decoder.make_self_mask(tgt_tokens_in)                                   # [B,1,T,T]
        cross_mask = self.decoder.make_cross_padding_mask(tgt_tokens_in, src_tokens, self.pad_id_src)  # [B,1,T,S]
        return self.decoder(tgt_tokens_in, memory, self_mask, cross_mask)

    def forward(self, src_tokens, tgt_tokens):
        memory, _ = self.encode(src_tokens)
        tgt_in = tgt_tokens[:, :-1]              
        dec_h = self.decode(tgt_in, memory, src_tokens)
        logits = self.generator(dec_h)           
        return logits
