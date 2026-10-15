# p1-gpt-from-scratch · Shaer-GPT (شاعر)

**Decoding Language Models · Phase 1 · Module 1** · Episodes 02–03

We build a small GPT from scratch, piece by piece, and teach it to write **classical Arabic poetry**: pre-Islamic, early Islamic and Umayyad verse, the tradition of Imru' al-Qais and the *Mu'allaqāt*.
Everything later in the series is a modification of this model.

> **Why Arabic poetry?** It's strictly metered (16 classical *buḥūr*) and every verse ends on the same rhyme letter (*qāfiya*), so you can *see* whether the model has learned structure. And Arabic is under-served by today's tokenizers. We measured it: GPT-4's tokenizer needs **3.1×** more tokens per character for Arabic than for English (see [CONCEPTS.md](CONCEPTS.md#-the-arabic-token-tax-we-measured-this)).

## Learn

| Resource | What to take from it |
|---|---|
| **[CONCEPTS.md](CONCEPTS.md)** | Every concept with idea, formula, our example, and how OpenAI, Google, Anthropic, Qwen and DeepSeek do it |
| Karpathy, [“Let's build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) | **Code along** this time: pause, type, run. Swap his Shakespeare for our Arabic |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) §3.2 | Scaled dot-product attention, multi-head attention |
| [`ndim-infra` glossary](https://github.com/ndim-labs/ndim-infra/blob/main/docs/GLOSSARY.md) | Matmul, VRAM, fp16/bf16 |

## Data

```bash
# from this folder, using the ndim-infra environment
uv run --project ../ndim-infra python data/prepare.py            # classical Arabic poetry -> data/input.txt
uv run --project ../ndim-infra python data/prepare.py --stats    # see all eras
uv run --project ../ndim-infra python data/prepare.py --strip-diacritics   # easier variant
```

| Corpus | Poems | Characters | Vocab (chars) |
|---|---|---|---|
| Classical Arabic, with diacritics (default) | 10,076 | 5.8M | 72 |
| Classical Arabic, diacritics stripped | 10,076 | 3.6M | 63 |
| Tiny Shakespeare (`data/shakespeare.txt`, sanity check) | — | 1.1M | 65 |

Source: [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar) (~254k poems from aldiwan.net, with poet, era and meter). Its licence isn't declared, so we use it for study only, and **the data is never committed** (it's gitignored).

**Windows:** set `PYTHONIOENCODING=utf-8` so Arabic prints correctly in the terminal.

## Build: write it yourself, step by step

Put code in `src/`. Each step is one file; each must run before you move on.

### Episode 02 · Language models & self-attention

| # | File | Build | Done when |
|---|---|---|---|
| 1 | `src/data.py` | Character tokenizer (`encode`/`decode`), 90/10 train/val split, `get_batch()` returning `(x, y)` of shape `(B, T)` | `decode(encode(s)) == s` on an Arabic verse; `y` is `x` shifted by one |
| 2 | `src/bigram.py` | Bigram model: `nn.Embedding(vocab, vocab)`, cross-entropy loss, `generate()` | Starting loss ≈ `ln(72) ≈ 4.28`; loss drops well below it after training |
| 3 | `src/attention_math.py` | The “mathematical trick”: averaging past tokens with a loop → matmul → masked softmax | All three versions give the same tensor (`torch.allclose`) |
| 4 | `src/head.py` | One self-attention head: `key`, `query`, `value`, causal mask, scale by `1/√head_size` | Output shape `(B, T, head_size)`; future tokens get zero weight |

### Episode 03 · The Transformer, built

| # | File | Build | Done when |
|---|---|---|---|
| 5 | `src/gpt.py` | Multi-head attention → feed-forward → residuals → LayerNorm → stacked `Block`s + position embeddings | Forward pass runs; parameter count ≈ 10.7M |
| 6 | `src/train.py` | Training loop with AdamW, periodic train/val loss, sample generation, **W&B logging** | (a) **Sanity check:** Shakespeare val loss ≈ 1.5 (Karpathy's number). (b) **Arabic:** val loss beats the bigram clearly, and samples keep line structure and a repeating rhyme |

**Config** (Karpathy's final): `batch_size=64, block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2, lr=3e-4`
On the 6 GB laptop, start with `batch_size=32` if you hit out-of-memory. Use **fp16** on the Colab T4 and **bf16** on the laptop.

**W&B:** `wandb.init(entity="ndim-labs", project="p1-gpt-from-scratch", name="<who>-<what>-<n>")`

## Experiments that create buzz

Pick one each and present it at the next episode:

1. **Does it learn the rhyme?** For 100 generated poems, measure how often every verse ends on the same letter.
2. **Diacritics on vs off:** train both variants. Which reaches lower loss per character? Which *sounds* better?
3. **Era style transfer:** train on pre-Islamic only, then on Abbasid only (`--eras`). Can a reader tell them apart?
4. **Attention maps:** plot one head's weights on a verse. Does a head look back at the previous verse's rhyme?
5. **The token tax chart:** reproduce the Arabic vs English tokenizer table in [CONCEPTS.md](CONCEPTS.md) and post it.

## Check your understanding (answer in the research log)

1. Why is the bigram model's starting loss about 4.28 on our data?
2. Why divide attention scores by `√head_size`? What happens if you don't?
3. What does the causal mask stop the model from doing, and why does training need it?
4. What do residual connections and LayerNorm each fix?
5. Show the parameter-count arithmetic for one `Block`.
6. Name one way Gemma, Qwen or DeepSeek differs from our model, and why they made that choice.

## Module done when

- [ ] `src/` steps 1–6 run end to end
- [ ] A W&B run in `ndim-labs/p1-gpt-from-scratch`: the Shakespeare sanity check plus one Arabic run
- [ ] A research-log entry in `ndim-infra` answering the 6 questions
- [ ] 5-minute demo at the next episode: generated verses plus one buzz experiment
