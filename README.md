# Transformer-from-scratch
Reimplementation of the Transformer architecture (Attention Is All You Need, Vaswani et al., 2017) in PyTorch with training on translation datasets, and experiments on model variants.

# Introduction
This project re-creates the legendary Transformer model exactly as proposed in the original paper — using the same architecture, hyperparameters, and training strategy as the “small Transformer” configuration.

I trained the model on an English → Spanish dataset (~127 MB from Tableau), running for 20 epochs, and observed surprisingly strong translation performance with clear, fluent outputs.

## The goal of this project is twofold:

    - Demonstrate a faithful reproduction of the seminal Transformer model.

    - Showcase its effectiveness on real translation tasks even with modest data and compute.
