\documentclass[11pt]{article}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{longtable}
\geometry{margin=1in}

\title{\textbf{transformer-from-scratch}}
\author{Reimplementation of the Transformer architecture (Attention Is All You Need, Vaswani et al., 2017) in PyTorch}
\date{}

\begin{document}
\maketitle

\section*{Overview}
This project re-creates the legendary Transformer model exactly as proposed in the original paper \emph{Attention Is All You Need} (Vaswani et al., 2017). It uses the same architecture, hyperparameters, and training strategy as the ``small Transformer'' configuration.  

The model was trained on an English $\rightarrow$ Spanish dataset ($\sim$127 MB from Tableau) for \textbf{20 epochs}, achieving surprisingly fluent and high-quality translations.  

Goals of the project:
\begin{itemize}
    \item Faithfully reproduce the Transformer architecture from the original paper.
    \item Demonstrate effectiveness on real translation tasks with modest data and compute.
\end{itemize}

\section*{Key Features}
\begin{itemize}
    \item Full Transformer encoder--decoder with multi-head attention and position-wise feed-forward layers.
    \item Sinusoidal positional encodings as in the original paper.
    \item Scaled dot-product attention with parallel multi-head mechanism.
    \item Residual connections and layer normalization in all sub-layers.
    \item Regularization with dropout and label smoothing.
    \item Training loop built from scratch (no external seq2seq frameworks).
    \item Greedy decoding for inference.
\end{itemize}

\section*{Training Details}
\begin{description}
    \item[Dataset:] $\sim$127 MB parallel corpus (English--Spanish) collected from Tableau.
    \item[Model size:] Transformer (base/small) configuration:
    \begin{itemize}
        \item 6 encoder layers + 6 decoder layers
        \item Embedding dimension = 512
        \item Feed-forward dimension = 2048
        \item Attention heads = 8
        \item Dropout = 0.1
    \end{itemize}
    \item[Training:] 20 epochs
    \item[Optimizer:] Adam with learning rate warm-up (4000 steps)
    \item[Results:] Produced very fluent translations, confirming the effectiveness of attention-only architectures.
\end{description}

\section*{Installation}
\begin{verbatim}
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch
pip install -r requirements.txt
\end{verbatim}

\section*{Usage}
\subsection*{Train the model}
\begin{verbatim}
python -m src.train \
  --data data/eng-spa.txt \
  --epochs 20 \
  --save checkpoints/
\end{verbatim}

\subsection*{Translate a sentence}
\begin{verbatim}
python -m src.predict "How are you today?"
# -> "¿Cómo estás hoy?"
\end{verbatim}

\section*{Project Structure}
\begin{verbatim}
transformer-from-scratch/
│── src/
│   ├── model.py              # Transformer architecture
│   ├── modules/              # Attention, feedforward, embeddings
│   ├── train.py              # Training loop
│   ├── predict.py            # Inference script
│   └── data.py               # Dataset loading + preprocessing
│
├── checkpoints/              # Saved model weights
├── data/                     # Parallel corpus
├── requirements.txt
└── README.md
\end{verbatim}

\section*{Example Translations}
\begin{longtable}{|p{6cm}|p{6cm}|}
\hline
\textbf{English} & \textbf{Spanish} \\
\hline
Good morning, my friend. & Buenos días, mi amigo. \\
\hline
The model works really well. & El modelo funciona muy bien. \\
\hline
I love learning languages. & Me encanta aprender idiomas. \\
\hline
\end{longtable}

\section*{Reference}
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., \& Polosukhin, I. (2017). \emph{Attention Is All You Need}. NeurIPS 2017.  
\href{https://arxiv.org/abs/1706.03762}{Paper link}

\section*{Acknowledgements}
Inspired by the groundbreaking work in \emph{Attention Is All You Need}.  
Special thanks to the open-source PyTorch community.

\end{document}

