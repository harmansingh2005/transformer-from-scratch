import torch
from src.model import Transformer, Encoder, Decoder, EncoderLayer, DecoderLayer  # ensure classes are in model.py

def test_transformer_forward():
    B, S, T = 2, 6, 5
    src_vocab, tgt_vocab = 50, 60
    d_model, h, N = 32, 4, 2
    PAD_SRC = 0
    PAD_TGT = 0
    BOS_TGT = 1

    # Fake src with some padding
    src = torch.tensor([
        [3, 11, 9, 4, 0, 0],
        [7, 8,  5, 2, 1, 0],
    ], dtype=torch.long)

    # Fake tgt with BOS + tokens + pad
    tgt = torch.tensor([
        [BOS_TGT, 9,  4,  0, 0],
        [BOS_TGT, 12, 5,  6, 0],
    ], dtype=torch.long)

    model = Transformer(
        src_vocab_size=src_vocab,
        tgt_vocab_size=tgt_vocab,
        d_model=d_model,
        num_layers_enc=N,
        num_layers_dec=N,
        num_heads=h,
        d_ff=4*d_model,
        dropout=0.0,
        pad_id_src=PAD_SRC,
        pad_id_tgt=PAD_TGT,
    )

    logits = model(src, tgt)   # [B, T-1, tgt_vocab]
    assert logits.shape == (B, tgt.size(1) - 1, tgt_vocab)
    print("✅ Transformer forward test passed")

if __name__ == "__main__":
    test_transformer_forward()
