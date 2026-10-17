# p1-gpt-from-scratch

**Decoding Language Models · Phase 1 · Foundations**

Phase 1 goes from *watching* a GPT being built, to *building* our own, to *adapting* a real pretrained model.

| Stage | Format | Content | Code |
|---|---|---|---|
| **Preparation** | Solo practice | [Karpathy video](https://www.youtube.com/watch?v=kCc8FmEb1nY), [Attention paper](https://arxiv.org/html/1706.03762v7), [Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing), [ng-video-lecture](https://github.com/karpathy/ng-video-lecture), [nanoGPT](https://github.com/karpathy/nanoGPT) | Karpathy's, Shakespeare |
| **1.1a** | Webinar, live coding | Data → bigram → attention trick → self-attention (paper §3.2) | Our improved notebook |
| **1.1b** | Webinar, live coding | Multi-head → Block → full GPT → training (paper §3.1, 3.3, 3.5) | Our improved notebook |
| **1.2** | Webinar + build | **Serious nanoGPT**: our own Shaer-GPT with nanoGPT-quality training code, nanoGPT as the answer key | Ours, Arabic |
| **1.3** | Webinar + build | Full fine-tune vs LoRA vs QLoRA on Qwen 0.5B: the bridge to Phase 2 | Hugging Face `peft` |
| **Gate 1** | Review | Shaer-GPT demo + the fine-tuning comparison table | — |

Concepts for every stage: **[CONCEPTS.md](CONCEPTS.md)**. Each idea comes with its formula, an example, and how OpenAI, Google, Anthropic, Qwen and DeepSeek do it.

## Quick links

| | Resource | Link |
|---|---|---|
| 🎥 | Karpathy, “Let's build GPT: from scratch, in code, spelled out” | https://www.youtube.com/watch?v=kCc8FmEb1nY |
| 📄 | Attention Is All You Need (Vaswani et al., 2017) | https://arxiv.org/html/1706.03762v7 |
| 📓 | Karpathy's Colab notebook from the video | https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing |
| 💻 | ng-video-lecture: the finished code from the video | https://github.com/karpathy/ng-video-lecture |
| 💻 | nanoGPT: the serious version, used in 1.2 | https://github.com/karpathy/nanoGPT |
| 📘 | Our concepts guide | [CONCEPTS.md](CONCEPTS.md) |

**Status:** 🟡 Preparation in progress. Stage 1.1a starts once preparation is complete.

---

## Preparation · solo practice

Do these in order. They're the same GPT at three levels of detail. Use Karpathy's code on Shakespeare, as-is: don't modify it, and don't copy it into this repo.

| # | Resource | What to do | Done when |
|---|---|---|---|
| 1 | Karpathy, [“Let's build GPT”](https://www.youtube.com/watch?v=kCc8FmEb1nY) (about 2 h) | Watch end to end, pausing where needed | You can sketch the model diagram from memory |
| 2 | [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) | Read §3 (Model Architecture), especially §3.2 (Attention) and Figure 1 | You can write `softmax(QKᵀ/√d_k)·V` and explain each term |
| 3 | [Colab notebook](https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing) | **File → Save a copy in Drive**; set Runtime → **T4 GPU**; run each cell alongside the matching part of the video | Every cell runs and you can say what each one does |
| 4 | [ng-video-lecture](https://github.com/karpathy/ng-video-lecture) | Run `bigram.py`, then `gpt.py`, on Colab (commands below) | `gpt.py` reaches val loss about 1.5 and prints Shakespeare-like text |
| 5 | [nanoGPT](https://github.com/karpathy/nanoGPT) | Read the README; skim `model.py` and `train.py` | You know which file does what (we use it properly in 1.2) |

**Running ng-video-lecture on Colab:**
```
!git clone https://github.com/karpathy/ng-video-lecture
%cd ng-video-lecture
!python bigram.py
!python gpt.py      # tens of minutes on a T4
```

**Note:** nanoGPT's README now says (Nov 2025) the repo is *deprecated* in favour of [nanochat](https://github.com/karpathy/nanochat). It's still the clearest code for learning how a GPT is trained, which is why we use it.

### Self-check: ready for 1.1a?

Answer without notes:
- [ ] What does the model predict, and what is `y` relative to `x`?
- [ ] Why is the bigram model's starting loss about `ln(65) ≈ 4.17`?
- [ ] Write the attention formula and explain Q, K, V, and the `√d_k`.
- [ ] Why does a decoder need the causal mask?
- [ ] What do multi-head attention, the feed-forward layer, residual connections and LayerNorm each add?
- [ ] `gpt.py` ran to about 1.5 val loss on your Colab

All ticked → tell us, and we start **1.1a**.

---

## 1.1a · Webinar: from data to self-attention

**Live-coded:** a cleaner, better-explained version of Karpathy's notebook, on Shakespeare so everyone can follow.
**Paper:** §3.2, scaled dot-product attention.

1. Data and character tokenizer → `get_batch`
2. Bigram language model → loss → generation
3. The "mathematical trick": loop → matmul → masked softmax
4. One self-attention head

*Details are prepared together before the session.*

## 1.1b · Webinar: the full Transformer

**Paper:** §3.1 (encoder/decoder stacks), §3.3 (feed-forward), §3.5 (positional encoding).

1. Multi-head attention
2. Feed-forward, residual connections, LayerNorm → `Block`
3. Full GPT
4. Training and sampling

*Details are prepared together before the session.*

---

## 1.2 · Webinar + build: Serious nanoGPT → Shaer-GPT (شاعر)

We write **our own** GPT with nanoGPT-quality training code (AdamW, warmup + cosine LR, gradient accumulation, fp16/bf16 autocast, checkpoints, W&B), with nanoGPT open as the answer key. We train it on **classical Arabic poetry**: pre-Islamic, early Islamic and Umayyad verse, the tradition of Imru' al-Qais and the *Mu'allaqāt*.

> **Why Arabic poetry?** It's strictly metered (16 classical *buḥūr*), and every verse ends on the same rhyme letter (*qāfiya*), so you can *see* whether the model learned structure. Arabic is also under-served by today's tokenizers: GPT-4's tokenizer needs **3.1×** more tokens per character for Arabic than for English ([we measured it](CONCEPTS.md#-the-arabic-token-tax-we-measured-this)).

### Data (ready)

```bash
# from this folder, using the ndim-infra environment
uv run --project ../ndim-infra python data/prepare.py                     # classical Arabic poetry -> data/input.txt
uv run --project ../ndim-infra python data/prepare.py --stats             # list all eras
uv run --project ../ndim-infra python data/prepare.py --strip-diacritics  # easier variant
```

| Corpus | Poems | Characters | Vocab (chars) |
|---|---|---|---|
| Classical Arabic, with diacritics (default) | 10,076 | 5.8M | 72 |
| Classical Arabic, diacritics stripped | 10,076 | 3.6M | 63 |

Source: [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar) (~254k poems from aldiwan.net). Its licence isn't declared, so it's for study only, and **the data is never committed** (it's gitignored). On Windows, set `PYTHONIOENCODING=utf-8`.

**Code:** `src/`, our own. `src/data.py` already has a skeleton with TODOs and a self-check.

**nanoGPT tips for our hardware:** on a Colab T4, pass `--dtype=float16`. nanoGPT picks bf16 whenever PyTorch reports it as "supported", but the T4 only emulates bf16, which is about 10× slower (see the `ndim-infra` Week 0 results). On the Windows laptop, pass `--compile=False`.

*Details are prepared together before the session.*

## 1.3 · Webinar + build: full fine-tune vs LoRA vs QLoRA

The bridge to Phase 2. These techniques aren't in Karpathy's material: they come from [LoRA](https://arxiv.org/abs/2106.09685) (Microsoft, 2021) and [QLoRA](https://arxiv.org/abs/2305.14314) (University of Washington, 2023).

The same small model (Qwen 0.5B) and the same dataset, three ways. Compare memory, speed and quality.
**Tools:** Hugging Face `transformers`, `peft`, `bitsandbytes`.

*Details are prepared together before the session.*

---

## Gate 1 · Review

- [ ] Shaer-GPT demo: generated verses plus one experiment
- [ ] Fine-tuning comparison table (full fine-tune vs LoRA vs QLoRA)
- [ ] Each person presents one component they built
