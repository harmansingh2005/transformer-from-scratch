import torch
from src.modules import PositionwiseFeedForward

def test_ffn():
    B, T, d_model, d_ff = 2, 4, 16, 64
    x = torch.randn(B, T, d_model)

    ffn = PositionwiseFeedForward(d_model, d_ff, dropout=0.0)
    out = ffn(x)

    # Check shape stays the same
    assert out.shape == (B, T, d_model)
    print("✅ PositionwiseFeedForward test passed")

if __name__ == "__main__":
    test_ffn()
