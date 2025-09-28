import torch
from src.model import DecoderLayer, Encoder, EncoderLayer  # Encoder used to make memory

def test_decoder_layer():
    B, T_src, T_tgt = 2, 6, 5
    vocab_size = 40
    d_model = 32
    h = 4
    N = 2
    pad_id = 0

    # Fake src tokens with some padding
    src = torch.tensor([
        [5, 9, 3, 1, 0, 0],
        [7, 2, 8, 4, 6, 0],
    ], dtype=torch.long)

    # Build a tiny encoder to produce memory
    enc = Encoder(vocab_size=vocab_size, d_model=d_model, num_layers=N,
                  num_heads=h, d_ff=4*d_model, dropout=0.0, pad_id=pad_id)
    src_mask = enc.make_src_padding_mask(src, pad_id)  # [B,1,S,S]
    memory = enc(src, src_mask)                        # [B,S,D]

    # Fake target tokens (just ids; padding at end)
    tgt = torch.tensor([
        [1, 3, 4, 0, 0],
        [1, 2, 5, 6, 0],
    ], dtype=torch.long)

    layer = DecoderLayer(d_model, h, d_ff=4*d_model, dropout=0.0)

    # Build masks
    look_ahead = layer.make_look_ahead_mask(T_tgt, device=tgt.device)   # [1,1,T,T]
    tgt_pad_mask = layer.make_tgt_padding_mask(tgt, pad_id)             # [B,1,T,T]
    self_attn_mask = look_ahead | tgt_pad_mask                          # broadcast to [B,1,T,T]

    cross_mask = layer.make_cross_padding_mask(tgt, src, pad_id)        # [B,1,T_dec,T_enc]
    y = torch.randn(B, T_tgt, d_model)
    out = layer(y, memory, self_attn_mask, cross_mask)

    assert out.shape == (B, T_tgt, d_model)
    print("DecoderLayer test passed !")

if __name__ == "__main__":
    test_decoder_layer()
