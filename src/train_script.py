import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import os
from .data import train_sentencepiece, make_dataloaders
from .model import Transformer

def load_pairs(src_file, tgt_file):
    print(f"Loading data from {src_file} and {tgt_file}")
    with open(src_file, encoding="utf-8") as f_src, open(tgt_file, encoding="utf-8") as f_tgt:
        src_lines = [l.strip() for l in f_src]
        tgt_lines = [l.strip() for l in f_tgt]
    print(f"Loaded {len(src_lines)} source and {len(tgt_lines)} target sentences")
    return list(zip(src_lines, tgt_lines))

# Load dataset
train_pairs = load_pairs("data/train_full.src", "data/train_full.tgt")
valid_pairs = load_pairs("data/valid_full.src", "data/valid_full.tgt")

batch_size = 32  
max_len = 128

train_loader, valid_loader, vocab_size = make_dataloaders(
    train_pairs, valid_pairs, "spm_shared.model",  # Use existing tokenizer
    batch_size=batch_size, max_len=max_len, shuffle=True
)

print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(valid_loader)}")

# sanity check on dataloader only
src, tgt = next(iter(train_loader))
print("src shape:", src.shape)
print("tgt shape:", tgt.shape)
print("vocab size:", vocab_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pad_id = 0  # SPECIALS["pad"]

model = Transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    d_model=512,
    num_layers_enc=6,
    num_layers_dec=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
).to(device)

with torch.no_grad():
    sb, tb = src[:4].to(device), tgt[:4].to(device)
    logits_sanity = model(sb, tb)  # [B, T-1, V]
print("SANITY — src:", sb.shape, "tgt:", tb.shape, "logits:", logits_sanity.shape)


criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR for stability
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # Removed verbose
grad_clip = 1.0
epochs = 20  

print(f"Training with {epochs} epochs, LR={optimizer.param_groups[0]['lr']}")

def run_epoch(loader, train_mode=True):
    """
    Train / validate loops with progress bars
    Run a single epoch of training or validation.
    """
    model.train() if train_mode else model.eval()
    total_loss, steps = 0.0, 0
    
    desc = "Training" if train_mode else "Validation"
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch_idx, (src, tgt) in enumerate(pbar):
        src, tgt = src.to(device), tgt.to(device)

        # Forward: model does teacher forcing internally with tgt[:, :-1]
        logits = model(src, tgt)         # [B, T-1, V]
        gold   = tgt[:, 1:]              # [B, T-1]

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            gold.reshape(-1)
        )

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        steps += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / steps:.4f}'
        })
        steps += 1

    return total_loss / max(steps, 1)

best_val = float("inf")
patience = 5
patience_counter = 0

# Create checkpoints directory
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(1, epochs + 1):
    print(f"\n--- Epoch {epoch}/{epochs} ---")
    start_time = time.time()
    
    train_loss = run_epoch(train_loader, train_mode=True)
    val_loss   = run_epoch(valid_loader, train_mode=False)
    
    # Update scheduler
    scheduler.step(val_loss)
    
    epoch_time = time.time() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
          f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")

    is_best = val_loss < best_val
    if is_best:
        best_val = val_loss
        patience_counter = 0
        ckpt_path = f"checkpoints/transformer_best_full.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "hparams": {
                "d_model": 512, "num_heads": 8, "d_ff": 2048,
                "num_layers_enc": 6, "num_layers_dec": 6,
                "dropout": 0.1, "pad_id": pad_id
            }
        }, ckpt_path)
        print(f"New version saved: {ckpt_path}")
    else:
        patience_counter += 1
        
    # Save regular checkpoint every few epochs
    if epoch % 5 == 0:
        regular_ckpt = f"checkpoints/transformer_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "hparams": {
                "d_model": 512, "num_heads": 8, "d_ff": 2048,
                "num_layers_enc": 6, "num_layers_dec": 6,
                "dropout": 0.1, "pad_id": pad_id
            }
        }, regular_ckpt)
        print(f"📁 Checkpoint saved: {regular_ckpt}")
    
    # Early stopping
    if patience_counter >= patience:
        print(f"⏹️  Early stopping triggered after {patience} epochs without improvement")
        break

print(f"\n🎯 Training completed! Best validation loss: {best_val:.4f}")
