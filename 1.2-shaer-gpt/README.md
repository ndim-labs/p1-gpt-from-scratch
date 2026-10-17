# 1.2 · Webinar + build: Serious nanoGPT → Shaer-GPT (شاعر)

We write **our own** GPT with nanoGPT-quality training code (AdamW, warmup + cosine LR, gradient accumulation, fp16/bf16 autocast, checkpoints, W&B), with nanoGPT open as the answer key. We train it on **classical Arabic poetry**: pre-Islamic, early Islamic and Umayyad verse, the tradition of Imru' al-Qais and the *Mu'allaqāt*.

> **Why Arabic poetry?** It's strictly metered (16 classical *buḥūr*), and every verse ends on the same rhyme letter (*qāfiya*), so you can *see* whether the model learned structure. Arabic is also under-served by today's tokenizers: GPT-4's tokenizer needs **3.1×** more tokens per character for Arabic than for English ([we measured it](../CONCEPTS.md#-the-arabic-token-tax-we-measured-this)).

### Data (ready)

```bash
# from this folder, using the ndim-infra environment
uv run --project ../../ndim-infra python data/prepare.py                     # classical Arabic poetry -> data/input.txt
uv run --project ../../ndim-infra python data/prepare.py --stats             # list all eras
uv run --project ../../ndim-infra python data/prepare.py --strip-diacritics  # easier variant
```

| Corpus | Poems | Characters | Vocab (chars) |
|---|---|---|---|
| Classical Arabic, with diacritics (default) | 10,076 | 5.8M | 72 |
| Classical Arabic, diacritics stripped | 10,076 | 3.6M | 63 |

Source: [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar) (~254k poems from aldiwan.net). Its licence isn't declared, so it's for study only, and **the data is never committed** (it's gitignored). On Windows, set `PYTHONIOENCODING=utf-8`.

**Code:** `src/` (run commands from this folder), our own. `src/data.py` already has a skeleton with TODOs and a self-check.

**nanoGPT tips for our hardware:** on a Colab T4, pass `--dtype=float16`. nanoGPT picks bf16 whenever PyTorch reports it as "supported", but the T4 only emulates bf16, which is about 10× slower (see the `ndim-infra` Week 0 results). On the Windows laptop, pass `--compile=False`.

*Details are prepared together before the session.*
