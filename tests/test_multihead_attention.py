import torch
from src.modules import MultiHeadAttention

def test_multihead_attention():
    B, T, d_model, h = 2, 5, 16, 4
    x = torch.randn(B, T, d_model)

    mha = MultiHeadAttention(d_model, h, dropout=0.0)
    out, attn = mha(x, x)

    # Output shape: [B, T, d_model]
    assert out.shape == (B, T, d_model)
    # Attention shape: [B, h, T, T]
    assert attn.shape == (B, h, T, T)
    # Row sums = 1
    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)

    print("✅ MultiHeadAttention test passed")

if __name__ == "__main__":
    test_multihead_attention()
