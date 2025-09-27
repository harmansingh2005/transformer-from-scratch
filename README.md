# Transformer-from-scratch
Reimplementation of the Transformer architecture (Attention Is All You Need, Vaswani et al., 2017) in PyTorch with training on translation datasets, and experiments on model variants.

# Introduction
This project re-creates the legendary Transformer model exactly as proposed in the original paper — using the same architecture, hyperparameters, and training strategy as the “small Transformer” configuration.

I trained the model on an English → Spanish dataset (~127 MB from Tableau), running for 20 epochs, and observed surprisingly strong translation performance with clear, fluent outputs.

## The goal of this project is twofold:

    - Demonstrate a faithful reproduction of the seminal Transformer model.
    - Showcase its effectiveness on real translation tasks even with modest data and compute.

# Key Features

- **Full Transformer encoder–decoder** with multi-head attention and position-wise feed-forward layers.
- **Positional encodings** implemented exactly as in the paper.
- **Scaled dot-product attention** with parallel multi-head mechanism.
- **Residual connections + layer normalization** on all sub-layers.
- **Label smoothing** and dropout regularization.
- **Training loop from scratch** (no external seq2seq frameworks).
- **Greedy decoding** for inference.

## Training Details

- **Dataset:** ~127 MB parallel corpus (English ↔ Spanish) collected from Tableau.
- **Model size:** identical to the Transformer (base/small) in the paper.
- **Layers:** 6 encoder + 6 decoder.
- **Embedding dim:** 512.
- **FFN inner dim:** 2048.
- **Attention heads:** 8.
- **Dropout:** 0.1.
- **Training:** 20 epochs.
- **Optimizer:** Adam with custom learning rate schedule (warm-up steps = 4000).
- **Hardware:** Trained on an NVIDIA RTX 3090 Ti GPU.
- **Loss & Optimization Setup:**
  ```python
  criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
  optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Lower LR for stability
  scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)  # No verbose
  grad_clip = 1.0
  epochs = 20

- **Results:** Achieved very fluent translations on test examples, confirming the power of attention-only architectures.

