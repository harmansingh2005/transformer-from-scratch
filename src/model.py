import math
import torch
import torch.nn as nn
from .modules import MultiheadAttention, FeedForward, ResidualConnection ,InputEmbedding, PositionalEncoding



class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.rc1 = ResidualConnection(d_model, dropout)
        self.rc2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask):
        x = self.rc1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.rc2(x, self.ffn)
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=10000, dropout=dropout)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.rc1 = ResidualConnection(d_model, dropout)
        self.rc2 = ResidualConnection(d_model, dropout)
        self.rc3 = ResidualConnection(d_model, dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x = self.rc1(x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.rc2(x, lambda x: self.cross_attn(x, memory, memory, cross_mask))
        x = self.rc3(x, self.ffn)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.embed = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=10000, dropout=dropout)

        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, self_mask, cross_mask):
        x = self.embed(tgt)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, memory, self_mask, cross_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, num_layers_enc=6, num_layers_dec=6,
                 num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_layers_enc, num_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers_dec, num_heads, d_ff, dropout)

        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        # weight tying with decoder embedding
        self.generator.weight = self.decoder.embed.embeddings.weight

    def make_pad_mask(self, tokens, pad_id):
        # True = mask, shape [B,1,1,T]
        return (tokens == pad_id).unsqueeze(1).unsqueeze(2)

    def make_look_ahead_mask(self, size, device):
        # upper-triangular mask
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1).unsqueeze(0).unsqueeze(0)

    def encode(self, src, pad_id):
        src_mask = self.make_pad_mask(src, pad_id)
        memory = self.encoder(src, src_mask)
        return memory, src_mask

    def decode(self, tgt, memory, src, pad_id_src, pad_id_tgt):
        self_mask = self.make_pad_mask(tgt, pad_id_tgt) | self.make_look_ahead_mask(tgt.size(1), tgt.device)
        cross_mask = self.make_pad_mask(src, pad_id_src)
        return self.decoder(tgt, memory, self_mask, cross_mask)

    def forward(self, src, tgt, pad_id_src=0, pad_id_tgt=0):
        memory, _ = self.encode(src, pad_id_src)
        tgt_in = tgt[:, :-1]  # remove last token for teacher forcing
        dec_out = self.decode(tgt_in, memory, src, pad_id_src, pad_id_tgt)
        logits = self.generator(dec_out)
        return logits
