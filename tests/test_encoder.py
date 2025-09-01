import torch
from src.model import Encoder, EncoderLayer 

def test_encoder_stack():
    B, T = 2, 7
    vocab_size = 50
    d_model = 32
    h = 4
    N = 3
    pad_id = 0

    # Fake batch with some padding at the end
    src = torch.tensor([
        [3, 11, 9, 4, 0, 0, 0],
        [7, 8,  5, 2, 1, 0, 0],
    ], dtype=torch.long)

    enc = Encoder(vocab_size=vocab_size, d_model=d_model, num_layers=N,
                  num_heads=h, d_ff=4*d_model, dropout=0.0, pad_id=pad_id)

    # Build padding mask
    src_mask = enc.make_src_padding_mask(src, pad_id)  # [B,1,T,T]
    out = enc(src, src_mask)  # [B, T, d_model]

    assert out.shape == (B, T, d_model)
    assert src_mask.shape == (B, 1, T, T)
    print("✅ Encoder stack test passed")

if __name__ == "__main__":
    test_encoder_stack()
