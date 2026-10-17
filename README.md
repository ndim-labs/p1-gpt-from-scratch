# p1-gpt-from-scratch

**Decoding Language Models · Phase 1 · Foundations**

Phase 1 goes from *understanding* a GPT, to *building* our own, to *adapting* a real pretrained model.

| Stage | Folder | Format | Content | Code |
|---|---|---|---|---|
| **1.0** | [`1.0-preparation/`](1.0-preparation/) | Live group session | **Basics + the Attention paper**, while everyone runs Karpathy's notebook (`gpt_dev.ipynb`) | Karpathy's, Shakespeare |
| **1.1a** | [`1.1a-data-to-attention/`](1.1a-data-to-attention/) | Webinar, live coding | Data → bigram → attention trick → self-attention (paper §3.2) | Our improved notebook |
| **1.1b** | [`1.1b-transformer/`](1.1b-transformer/) | Webinar, live coding | Multi-head → Block → full GPT → training (paper §3.1, 3.3, 3.5) | Our improved notebook |
| **1.2** | [`1.2-shaer-gpt/`](1.2-shaer-gpt/) | Webinar + build | **Serious nanoGPT**: our own Shaer-GPT (شاعر) on classical Arabic poetry, nanoGPT as the answer key | Ours, Arabic |
| **1.3** | [`1.3-lora-qlora/`](1.3-lora-qlora/) | Webinar + build | Full fine-tune vs LoRA vs QLoRA on Qwen 0.5B: the bridge to Phase 2 | Hugging Face `peft` |
| **Gate 1** | — | Review | Shaer-GPT demo + fine-tuning comparison table | — |

**Status:** ✅ Organizer's solo preparation complete · ✅ 1.0 notebook annotated, with [charts](1.0-preparation/README.md#what-youll-see-charts-from-the-notebook) · 🟡 next: **1.0 session**

Concepts for every stage: **[CONCEPTS.md](CONCEPTS.md)**. Each idea comes with its formula, an example, and how OpenAI, Google, Anthropic, Qwen and DeepSeek do it.

## Quick links

| | Resource | Link |
|---|---|---|
| 📓 | **1.0 notebook** (Karpathy's `gpt_dev.ipynb`, in this repo) | [Open in Colab](https://colab.research.google.com/github/ndim-labs/p1-gpt-from-scratch/blob/main/1.0-preparation/gpt_dev.ipynb) |
| 🎥 | Karpathy, “Let's build GPT: from scratch, in code, spelled out” | https://www.youtube.com/watch?v=kCc8FmEb1nY |
| 📄 | Attention Is All You Need (Vaswani et al., 2017) | https://arxiv.org/html/1706.03762v7 |
| 📓 | Karpathy's original Colab notebook | https://colab.research.google.com/drive/1JMLa53HDuA-i7ZBmqV7ZnA3c_fvtXnx-?usp=sharing |
| 💻 | ng-video-lecture: the finished code from the video | https://github.com/karpathy/ng-video-lecture |
| 💻 | nanoGPT: the serious version, used in 1.2 | https://github.com/karpathy/nanoGPT |

**Note:** nanoGPT's README now says (Nov 2025) the repo is *deprecated* in favour of [nanochat](https://github.com/karpathy/nanochat). It's still the clearest code for learning how a GPT is trained, which is why we use it.

## Repo layout

```
p1-gpt-from-scratch/
├── CONCEPTS.md                 # concepts guide for all stages
├── 1.0-preparation/            # session plan + gpt_dev.ipynb (annotated) + images/ (charts)
├── 1.1a-data-to-attention/     # live-coding notebook (to be prepared)
├── 1.1b-transformer/           # live-coding notebook (to be prepared)
├── 1.2-shaer-gpt/              # our own code: data/ (Arabic poetry prep) + src/
└── 1.3-lora-qlora/             # fine-tuning comparison (to be prepared)
```

## Gate 1 · Review

- [ ] Shaer-GPT demo: generated verses plus one experiment
- [ ] Fine-tuning comparison table (full fine-tune vs LoRA vs QLoRA)
- [ ] Each person presents one component they built
