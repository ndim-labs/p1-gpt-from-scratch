# 1.0 · Preparation session: basics + the Attention paper + the notebook

**Format:** live group session (about 2 h). Concepts and the paper are presented while everyone runs the notebook alongside.
**Goal:** everyone leaves with the basics, a working mental model of attention, and a notebook that has run end to end. That's the foundation for 1.1a onward.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ndim-labs/p1-gpt-from-scratch/blob/main/1.0-preparation/gpt_dev.ipynb) ← **`gpt_dev.ipynb`**, Karpathy's companion notebook to the video

## Before the session (participants)

1. Open the notebook with the badge above, then **File → Save a copy in Drive**.
2. **Runtime → Change runtime type → T4 GPU**.
3. Optional: watch Karpathy's [“Let's build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) and skim [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) §3.

## Session flow

Each block: **concept → paper → run the notebook cells**.

| # | Time | Block | Paper | Notebook section | [CONCEPTS.md](../CONCEPTS.md) |
|---|---|---|---|---|---|
| 1 | 10 min | **What a language model does:** next-token prediction; the big picture from text to tokens to the model to the next token | Abstract, §1 | — | §0 |
| 2 | 15 min | **Data and tokens:** character tokenizer, `encode`/`decode`, train/val split | — | Dataset download → tokenizer → train/val split | §1 |
| 3 | 15 min | **Batches:** context (`block_size`), why `y` is `x` shifted by one, batch dimension | — | `get_batch` | §11 |
| 4 | 15 min | **The simplest model:** bigram, logits, cross-entropy, why the starting loss ≈ ln(65) ≈ 4.17, sampling | — | Bigram language model + training loop | §10, §11 |
| 5 | 25 min | **The mathematical trick:** averaging the past with a loop → matmul → masked softmax | §3.2 | "The mathematical trick in self-attention" | §4, §5 |
| 6 | 25 min | **Self-attention:** Q, K, V; `softmax(QKᵀ/√d_k)·V`; why √d_k; causal mask; attention as communication | **§3.2.1**, Figure 2 (left) | Self-attention head + notes | §4, §5 |
| 7 | 10 min | **The rest, previewed:** multi-head attention, feed-forward, residual connections, LayerNorm, positions: the full block | §3.1, 3.2.2, 3.3, 3.5, **Figure 1** | LayerNorm + full finished code | §6–9 |
| 8 | 5 min | **Wrap-up:** what's next in 1.1a/1.1b, homework | — | — | — |

**Presenter tip:** the cells that train models take minutes. Run them before the session, or start them at the beginning of a block and talk while they train.

## What you'll see: charts from the notebook

These come from a full run of [`gpt_dev.ipynb`](gpt_dev.ipynb). The notebook saves the same charts in its outputs.

### 1 · Bigram model: training vs validation loss
![Bigram loss](images/1_bigram_loss.png)
Loss falls from about 4.7 and gets stuck at **about 2.5**. Training and validation stay together, so the model isn't memorizing: it's just too simple. With one character of memory, it can't form words.

### 2 · Averaging weights: "the past only, equal weights"
![Averaging weights](images/2_averaging_weights.png)
Row *t* spreads its weight equally over positions 0…*t*. Zeros above the diagonal mean no peeking at the future.

### 3 · Self-attention weights
![Attention weights](images/3_attention_weights.png)
Same triangle, but now the weights **depend on the data**: some past positions matter much more than others. This is attention (paper §3.2.1).

### 4 · Why divide by √d
![Softmax scaling](images/4_softmax_scaling.png)
Small scores give a spread-out softmax (left); large scores give nearly one-hot (right), and learning stalls. Dividing by √d_k keeps the scores small.

### 5 · Full GPT: training vs validation loss
![GPT train vs val loss](images/5_gpt_train_val_loss.png)
The GPT (4 layers, 4 heads, 0.21M parameters) breaks through the bigram's 2.5 limit and reaches **val loss about 1.82** in 5,000 steps. The growing gap between the curves is the first sign of memorizing. Karpathy's full-size `gpt.py` (10.7M parameters, dropout 0.2) reaches about 1.48.

## Paper reading map: Attention Is All You Need

| Section | What to take from it | Covered in |
|---|---|---|
| Abstract, §1–2 | Why attention replaced RNNs: parallelism + long-range connections | 1.0 block 1 |
| **§3.1** Encoder and decoder stacks, **Figure 1** | The block diagram. GPT uses **only the decoder** side, without cross-attention | 1.0 block 7, 1.1b |
| **§3.2.1** Scaled dot-product attention | The core formula and the `√d_k` scaling | 1.0 block 6, 1.1a |
| §3.2.2 Multi-head attention | Several heads in parallel, then concatenate | 1.1b |
| §3.2.3 Applications of attention | Masking in the decoder ("prevent leftward information flow") | 1.0 block 6 |
| §3.3 Position-wise feed-forward | The per-token MLP | 1.1b |
| §3.4 Embeddings and softmax | Shared embedding / output weights (weight tying) | 1.1b, 1.2 |
| §3.5 Positional encoding | Sinusoidal positions; GPT uses learned ones; modern models use RoPE | 1.1b |
| §4 Why self-attention | Cost comparison vs RNN/CNN | Optional |
| §5–6 Training, results | Adam, warmup schedule, label smoothing | 1.2 |

## Homework after the session

- [ ] Notebook runs end to end in your own Colab copy
- [ ] Answer without notes:
  1. What is `y` relative to `x`, and why?
  2. Why is the bigram model's starting loss about `ln(65) ≈ 4.17`?
  3. Write the attention formula and explain Q, K, V, and the `√d_k`.
  4. Why does a decoder need the causal mask?
  5. What do multi-head attention, the feed-forward layer, residual connections and LayerNorm each add?
- [ ] Optional: run [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) `gpt.py` to val loss about 1.5, and read the [nanoGPT](https://github.com/karpathy/nanoGPT) README

**Ready for 1.1a** when the notebook runs and you can answer all five.
