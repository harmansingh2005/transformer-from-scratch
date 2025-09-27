import torch
from src.modules import ScaledDotProductAttention

def test_scaled_dot_product_attention():
    """
    B → Batch size
    h → Number of heads
    Tq → Query length
    Tk → Key (and Value) length
    d_k → Key/Query depth per head
    
    """
    
    B, h, Tq, Tk, dk = 2, 3, 4, 5, 8
    # assigning random tensor for testing
    Q = torch.randn(B, h, Tq, dk) 
    K = torch.randn(B, h, Tk, dk)
    V = torch.randn(B, h, Tk, dk)

    attn = ScaledDotProductAttention(dk, dropout=0.0)
    out, w = attn(Q, K, V)

    assert out.shape == (B, h, Tq, dk)
    assert w.shape == (B, h, Tq, Tk)
    row_sums = w.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    print("ScaledDotProductAttention test passed!!")

if __name__ == "__main__":
    test_scaled_dot_product_attention()