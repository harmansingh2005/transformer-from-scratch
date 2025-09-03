import os
import tempfile
import torch

from src.data import (
    train_sentencepiece,
    make_dataloaders,
    SPECIALS,
)

def test_data_pipeline_end_to_end():
    # 1)parallel data
    train_pairs = [
        ("i like cats",          "ich mag katzen"),
        ("you like dogs",        "du magst hunde"),
        ("cats are cute",        "katzen sind süß"),
        ("dogs are smart",       "hunde sind klug"),
        ("i love nlp",           "ich liebe nlp"),
    ]
    valid_pairs = [
        ("i like dogs",          "ich mag hunde"),
        ("cats are smart",       "katzen sind klug"),
    ]

    # 2) train a tiny SPM model in a temp dir
    with tempfile.TemporaryDirectory() as td:
        src_path = os.path.join(td, "src.txt")
        tgt_path = os.path.join(td, "tgt.txt")
        with open(src_path, "w", encoding="utf-8") as f:
            for s, _ in train_pairs + valid_pairs:
                f.write(s.strip() + "\n")
        with open(tgt_path, "w", encoding="utf-8") as f:
            for _, t in train_pairs + valid_pairs:
                f.write(t.strip() + "\n")

        spm_prefix = os.path.join(td, "spm_shared")
        # vocab is fine for toy data
        train_sentencepiece([src_path, tgt_path], model_prefix=spm_prefix, vocab_size=42)

        spm_model_path = spm_prefix + ".model"

        # 3) build loaders 
        train_loader, valid_loader, vocab_size = make_dataloaders(
            train_pairs, valid_pairs, spm_model_path, batch_size=3, max_len=16, shuffle=False
        )

        assert vocab_size >= 10, "vocab_size seems unexpectedly small"

        # 4) fetch one batch and check tensors 
        src, tgt = next(iter(train_loader))  # tensors [B, T]
        assert isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor)
        assert src.dtype == torch.long and tgt.dtype == torch.long
        assert src.ndim == 2 and tgt.ndim == 2, "batches must be [B, T]"

        B, Ts = src.shape
        _, Tt = tgt.shape
        assert B == 3, "batch size mismatch"

        assert torch.all(src[:, 0] == SPECIALS["bos"])
        assert torch.all(tgt[:, 0] == SPECIALS["bos"])

        assert SPECIALS["pad"] == 0
        assert (src == SPECIALS["pad"]).sum().item() >= 0
        assert (tgt == SPECIALS["pad"]).sum().item() >= 0

        assert (src == SPECIALS["eos"]).any() and (tgt == SPECIALS["eos"]).any()

        print("data pipeline ok:",
              f"src shape={tuple(src.shape)} tgt shape={tuple(tgt.shape)} vocab_size={vocab_size}")
        
if __name__ == "__main__":
    test_data_pipeline_end_to_end()
