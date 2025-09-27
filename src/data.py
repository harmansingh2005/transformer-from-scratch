import torch
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Special token ids used everywhere (SPM + model)
SPECIALS = {"pad": 0, "bos": 1, "eos": 2, "unk": 3}

# -------------------- sentencepiece training --------------------
def train_sentencepiece(corpus_paths, model_prefix, vocab_size=8000, character_coverage=1.0):
    """
    Train a shared SentencePiece model on the given text files.
    Writes model_prefix.model and model_prefix.vocab
    """
    spm.SentencePieceTrainer.Train(
        input=",".join(corpus_paths),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        pad_id=SPECIALS["pad"],
        bos_id=SPECIALS["bos"],
        eos_id=SPECIALS["eos"],
        unk_id=SPECIALS["unk"],
    )

# -------------------- dataset --------------------
class ParallelTextDataset(Dataset):
    """
    Holds aligned pairs of (src_text, tgt_text) and encodes them with SPM.
    Adds BOS/EOS and truncates to max_len.
    """
    def __init__(self, lines_src, lines_tgt, spm_model_path, max_len=128):
        assert len(lines_src) == len(lines_tgt), "src/tgt sizes must match"
        self.src = lines_src
        self.tgt = lines_tgt
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        self.vocab_size = self.sp.vocab_size()
        self.max_len = max_len

    def encode(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = [SPECIALS["bos"]] + ids + [SPECIALS["eos"]]
        return ids[: self.max_len]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.encode(self.src[i]), self.encode(self.tgt[i])

# -------------------- batching utils --------------------
def pad_batch(seqs, pad_id=SPECIALS["pad"]):
    """
    seqs: list of python lists of token ids
    returns: LongTensor [B, T_max] padded with pad_id
    """
    tensors = [torch.tensor(s, dtype=torch.long) for s in seqs]
    return pad_sequence(tensors, batch_first=True, padding_value=pad_id)

def collate_parallel(batch):
    """
    batch: list of (src_ids_list, tgt_ids_list)
    returns: (src_tensor[B,T], tgt_tensor[B,T])
    """
    src, tgt = zip(*batch)           # tuples of python lists
    return pad_batch(src), pad_batch(tgt)

# -------------------- dataloaders --------------------
def make_dataloaders(train_pairs, valid_pairs, spm_model_path, batch_size=64, max_len=128, shuffle=True):
    """
    train_pairs / valid_pairs: list of (src_text, tgt_text)
    spm_model_path: path to .model file
    returns: (train_loader, valid_loader, vocab_size)
    """
    tr_ds = ParallelTextDataset(
        [s for s, _ in train_pairs],
        [t for _, t in train_pairs],
        spm_model_path,
        max_len
    )
    va_ds = ParallelTextDataset(
        [s for s, _ in valid_pairs],
        [t for _, t in valid_pairs],
        spm_model_path,
        max_len
    )

    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_parallel)
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_parallel)

    return tr, va, tr_ds.vocab_size
