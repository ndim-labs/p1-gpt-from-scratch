# p1-gpt-from-scratch

**Decoding Language Models · Phase 1 · Module 1** — Episodes 02–03

Build a small GPT, piece by piece, and train it on Shakespeare. Everything later in the series is a modification of this.

## Learn

| Resource | What to take from it |
|---|---|
| Karpathy, [“Let's build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) | **Code along** this time: pause, type, run |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §3.2 | Scaled dot-product attention, multi-head attention |
| [`ndim-infra` glossary](https://github.com/ndim-labs/ndim-infra/blob/main/docs/GLOSSARY.md) | Matmul, VRAM, fp16/bf16 |

## Setup

```bash
python data/prepare.py      # downloads Tiny Shakespeare to data/input.txt (not committed)
```

Use the `ndim-infra` environment (it already has torch + wandb). Locally: run from `ndim-infra` with `uv run`; on Colab: the bootstrap notebook.

## Build: write it yourself, step by step

Put code in `src/`. Each step is one file; each must run before you move on.

### Episode 02 · Language models & self-attention

| # | File | Build | Done when |
|---|---|---|---|
| 1 | `src/data.py` | Character tokenizer (`encode`/`decode`), 90/10 train/val split, `get_batch()` returning `(x, y)` of shape `(B, T)` | `decode(encode(s)) == s`; `y` is `x` shifted by one |
| 2 | `src/bigram.py` | Bigram model: `nn.Embedding(vocab, vocab)`, cross-entropy loss, `generate()` | Starting loss ≈ `ln(65) ≈ 4.17`; trained val loss ≈ 2.5 |
| 3 | `src/attention_math.py` | The “mathematical trick”: averaging past tokens with a loop → matmul → masked softmax | All three versions give the same tensor (`torch.allclose`) |
| 4 | `src/head.py` | One self-attention head: `key`, `query`, `value`, causal mask, scale by `1/√head_size` | Output shape `(B, T, head_size)`; future tokens get zero weight |

### Episode 03 · The Transformer, built

| # | File | Build | Done when |
|---|---|---|---|
| 5 | `src/gpt.py` | Multi-head attention → feed-forward → residuals → LayerNorm → stacked `Block`s + position embeddings | Model runs a forward pass; prints parameter count |
| 6 | `src/train.py` | Training loop with AdamW, periodic train/val loss, sample generation, **W&B logging** | Val loss ≈ **1.5** with Karpathy's config; samples look Shakespeare-ish |

**Config** (Karpathy's final): `batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2, lr=3e-4`
≈ 10.7M parameters. On the 6 GB laptop, start with `batch_size=32` if you hit out-of-memory.

**W&B:** `wandb.init(entity="ndim-labs", project="p1-gpt-from-scratch", name="<who>-<what>-<n>")`

## Check your understanding (answer in the research log)

1. Why is the bigram model's starting loss about 4.17?
2. Why divide attention scores by `√head_size`? What happens if you don't?
3. What does the causal mask stop the model from doing, and why does training need it?
4. What do residual connections and LayerNorm each fix?
5. Your parameter count: show the arithmetic for one `Block`.

## Module done when

- [ ] `src/` steps 1–6 run end to end
- [ ] A W&B run in `ndim-labs/p1-gpt-from-scratch` with val loss ≤ 1.6
- [ ] A research-log entry in `ndim-infra` answering the 5 questions
- [ ] 5-minute demo at the next episode: show generated text and explain one component
