
import os
import io
import random
import torch
import sentencepiece as spm # type: ignore
from torch.utils.data import Dataset, DataLoader

SPECIALS = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}

def train_sentencepiece(corpus_paths, model_prefix, vocab_size=8000, character_coverage=1.0):
    # Train a shared SPM model on concatenated corpora
    input_str = ",".join(corpus_paths)
    spm.SentencePieceTrainer.Train(
        input=input_str,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        pad_id=SPECIALS["pad"],
        bos_id=SPECIALS["bos"],
        eos_id=SPECIALS["eos"],
        unk_id=SPECIALS["unk"],
        user_defined_symbols=[]
    )

class ParallelTextDataset(Dataset):
    """
    Generic parallel data loader using a shared SentencePiece model.
    lines_src/lines_tgt are lists of strings aligned by index.
    """
    def __init__(self, lines_src, lines_tgt, spm_model_path, max_len=128):
        assert len(lines_src) == len(lines_tgt)
        self.src = lines_src
        self.tgt = lines_tgt
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.vocab_size = self.sp.vocab_size()
        self.max_len = max_len

    def encode(self, s):
        ids = self.sp.encode(s, out_type=int)
        ids = [SPECIALS["bos"]] + ids + [SPECIALS["eos"]]
        return ids[: self.max_len]

    def __len__(self): return len(self.src)

    def __getitem__(self, i):
        return self.encode(self.src[i]), self.encode(self.tgt[i])

def pad_batch(seqs, pad_id=SPECIALS["pad"]):
    max_len = max(len(s) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out

def collate_parallel(batch):
    src, tgt = zip(*batch)
    return pad_batch(src), pad_batch(tgt)

def make_dataloaders(train_pairs, valid_pairs, spm_model_path, batch_size=64, max_len=128, shuffle=True):
    tr_ds = ParallelTextDataset([s for s,_ in train_pairs], [t for _,t in train_pairs], spm_model_path, max_len)
    va_ds = ParallelTextDataset([s for s,_ in valid_pairs], [t for _,t in valid_pairs], spm_model_path, max_len)
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_parallel)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_parallel)
    return tr, va, tr_ds.vocab_size
