import torch
from src.model import EncoderLayer

def test_encoder_layer():
    B, T, d_model, h = 2, 6, 16, 4
    x = torch.randn(B, T, d_model)

    layer = EncoderLayer(d_model, h, d_ff=32, dropout=0.0)
    out = layer(x)

    assert out.shape == (B, T, d_model)
    print("EncoderLayer test passed!")

if __name__ == "__main__":
    test_encoder_layer()
