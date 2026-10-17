# Concepts: how a GPT works, and how the big labs build theirs

**Decoding Language Models · Phase 1 · Module 1**

Every concept below has four parts:
1. **Idea:** what it is, in plain words
2. **Formula:** the maths you'll implement
3. **In our model:** what it looks like in Shaer-GPT, our Arabic poetry model
4. **In the frontier models:** how OpenAI, Google, Anthropic and Chinese labs (Alibaba Qwen, DeepSeek, Moonshot Kimi) do it, according to their public papers and model cards

> **About sources.** Numbers come from each lab's published papers and model cards (linked at the end). OpenAI hasn't published GPT-4/5 architecture, and **Anthropic hasn't published Claude's architecture**. For those we say so, and use what they *have* published. Anthropic's interpretability research is the best public explanation of what happens *inside* a transformer. Double-check any number before quoting it publicly.

---

## 0. The big picture

A GPT does one thing: **given some text, predict the next token.** Repeat that, and it writes.

```
text → tokens → embeddings (+ positions) → [ Block × N ] → final norm → logits → softmax → next token
                                            Block = attention + feed-forward, each wrapped in norm + residual
```

Every model in this guide (GPT-2, Gemma, Qwen, DeepSeek, and presumably Claude) is a variation on this diagram. The differences are in the details, which is what the rest of this guide covers.

---

## 1. Tokenization

**Idea:** models read numbers, not text. A *tokenizer* splits text into pieces (tokens) and maps each to an integer ID. The *vocabulary* is the set of all possible tokens.

- **Character-level** (what we build first): one token per character. Tiny vocabulary, long sequences.
- **BPE (Byte-Pair Encoding)** (Module 2): start from bytes and repeatedly merge the most frequent pair into a new token. Large vocabulary, short sequences.

**Formula:** `encode: str → list[int]`, `decode: list[int] → str`, with `decode(encode(s)) == s`

**In our model:** classical Arabic poetry has **72 unique characters**, or 63 with the diacritics (ḥarakāt) stripped.
`"قفا نبك"` → `[50, 49, 28, 1, 54, 29, 51]`, one ID per letter and space (IDs from the sorted 72-character vocabulary, undiacritized spelling).

**In the frontier models:**

| Model | Lab | Tokenizer | Vocabulary |
|---|---|---|---|
| GPT-2 | OpenAI | Byte-level BPE (`r50k`) | 50,257 |
| GPT-4 | OpenAI | BPE `cl100k_base` | ~100k |
| GPT-4o, gpt-oss | OpenAI | BPE `o200k_base` / `o200k_harmony` | ~200k |
| Gemma 1/2 | Google | SentencePiece | 256k |
| Gemma 3 | Google | SentencePiece | 262k |
| Claude | Anthropic | Not published | — |
| Qwen2.5 / Qwen3 | Alibaba | Byte-level BPE | ~151k |
| DeepSeek-V3 | DeepSeek | Byte-level BPE | ~128k |
| Llama 3 | Meta | BPE (tiktoken-based) | 128k |

### 🔥 The Arabic token tax (we measured this)

We tokenized 200k characters of our Arabic poetry and 200k characters of Shakespeare:

| Tokenizer | English chars/token | Arabic chars/token | Arabic costs… |
|---|---|---|---|
| GPT-2 (OpenAI) | 3.29 | 0.90 | **3.7×** more tokens |
| GPT-4 `cl100k` (OpenAI) | 3.67 | 1.18 | **3.1×** |
| Qwen2.5 (Alibaba) | 3.67 | 1.41 | **2.6×** |
| GPT-4o `o200k` (OpenAI) | 3.73 | 1.66 | **2.2×** |
| DeepSeek-V3 | 3.69 | 1.72 | **2.2×** |

**Why it matters:** tokens are what you pay for, what fills the context window, and what the model computes over. Fully vowelled classical Arabic is a worst case, but the pattern holds: tokenizers trained mostly on English make Arabic speakers pay more for less context. Bigger, more multilingual vocabularies (o200k, DeepSeek) shrink the tax. *Reproduce it in Module 2 and post the chart.*

---

## 2. Token embeddings

**Idea:** each token ID becomes a vector of learned numbers, its "meaning coordinates". Similar tokens end up close together.

**Formula:** `x = E[token_id]`, where `E` is a `(vocab_size × d_model)` lookup table learned during training.

**In our model:** `nn.Embedding(72, 384)`: 72 characters × 384 numbers = 27,648 parameters.

**In the frontier models:**

| Model | d_model (vector size) |
|---|---|
| Original Transformer (Google, 2017) | 512 |
| GPT-2 small (OpenAI) | 768 |
| GPT-3 175B (OpenAI) | 12,288 |
| Qwen2.5-0.5B (Alibaba) | 896 |
| DeepSeek-V3 | 7,168 |

**Weight tying:** GPT-2 and Gemma reuse the embedding matrix `E` as the output layer, which saves `vocab × d_model` parameters. With a 256k vocabulary, that's a lot.

---

## 3. Positional information

**Idea:** attention on its own doesn't know word order: "the cat sat" and "sat cat the" look identical. Position has to be injected.

**Formulas:**
- **Learned absolute** (GPT-2, and ours): `x = tok_emb[id] + pos_emb[position]`
- **Sinusoidal** (original Transformer): `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(…)`
- **RoPE (Rotary Position Embedding):** rotate each query/key pair by an angle proportional to its position, `θ = pos × base^(−2i/d)`. Attention then depends on *relative* distance.

**In our model:** learned absolute positions, `nn.Embedding(256, 384)` for a 256-character context.

**In the frontier models:**

| Model | Position method | Context length |
|---|---|---|
| Original Transformer (Google) | Sinusoidal | — |
| GPT-2 / GPT-3 (OpenAI) | Learned absolute | 1,024 / 2,048 |
| gpt-oss (OpenAI) | RoPE (+ YaRN extension) | 128k |
| PaLM, Gemma (Google) | RoPE | Gemma 3: 128k |
| Claude (Anthropic) | Not published | 200k (public product spec) |
| Qwen2.5 / Qwen3 (Alibaba) | RoPE | 32k–128k |
| DeepSeek-V3 | RoPE (decoupled, inside MLA) | 128k |
| Llama 3 (Meta) | RoPE, base 500,000 | 128k |

**Takeaway:** almost every modern open model uses **RoPE**. Learned positions (ours) can't go beyond the length they were trained on.

---

## 4. Self-attention

**Idea:** each token asks a question (**query**). Every token offers a label (**key**) and some content (**value**). A token gathers the values of the tokens whose keys best match its query. This is how "it" finds out what "it" refers to.

**Formula** (scaled dot-product attention, Vaswani et al. 2017, §3.2.1):

```
Q = X·W_q,   K = X·W_k,   V = X·W_v
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k  + mask ) · V
```

- `Q·Kᵀ`: how well each query matches each key (a `T × T` score table)
- `/ √d_k`: keeps scores from growing with dimension. Without it, softmax saturates and gradients vanish.
- `softmax`: turns scores into weights that sum to 1

**In our model:** in the verse *قِفا نَبكِ مِن ذِكرى حَبيبٍ وَمَنزِلِ* ("Halt, let us weep at the memory of a beloved and an abode", the opening of Imru' al-Qais's *Mu'allaqa*), the head predicting the rhyme letter at the end should attend strongly to the matching rhyme (*qāfiya*) of the previous verse. Plot your attention weights and look for this.

**In the frontier models:** every model in this guide uses this exact formula at its core. They differ in *how many* keys and values they keep (section 6) and *which* tokens each one can see (section 5).

**From Anthropic's research:** *A Mathematical Framework for Transformer Circuits* (2021) splits each head into a **QK circuit** (*where* to look) and an **OV circuit** (*what* to copy). *In-context Learning and Induction Heads* (2022) found **induction heads**: pairs of heads that find "the last time this token appeared" and copy what came next. That's a core mechanism behind in-context learning, and your model will likely grow some. Our poetry corpus is full of repeated rhymes and formulas to test this on.

---

## 5. The causal mask

**Idea:** when predicting token *t*, the model must not see tokens *t+1, t+2, …*, or it would just copy the answer. So future positions get a score of −∞, which becomes 0 after softmax.

**Formula:** `mask[i][j] = 0 if j ≤ i else −∞`, a lower-triangular matrix (`torch.tril`)

**In our model:** while reading ق-ف-ا, the model predicting the 3rd letter sees only ق and ف.

**In the frontier models:**

| Model | Lab | Who can see whom |
|---|---|---|
| GPT-2, Qwen, DeepSeek, Llama | — | Full causal: every past token |
| GPT-3 / gpt-oss | OpenAI | Alternate **dense** layers and **locally banded** (sliding-window) layers |
| Gemma 2 | Google | Alternate local (4,096-token window) and global layers, 1:1 |
| Gemma 3 | Google | 5 local (1,024-token window) : 1 global. Cuts memory for long contexts |

**Sliding-window attention** = a causal mask that also blocks tokens further back than *W*. It's cheaper, and the global layers still connect everything.

---

## 6. Multi-head attention and its descendants (MHA → MQA → GQA → MLA)

**Idea:** run several attention heads in parallel, each with its own Q/K/V, so different heads can track different things (rhyme, grammar, meter). Then concatenate the results.

**Formula:** `MultiHead(X) = Concat(head₁, …, head_h) · W_o`, where `d_head = d_model / h`

**The memory problem:** at generation time the model caches every past token's K and V (the **KV cache**). With long contexts that cache dominates GPU memory, so the labs found ways to shrink it:

| Variant | Idea | Used by |
|---|---|---|
| **MHA:** Multi-Head | Every head has its own K and V | Original Transformer, GPT-2, GPT-3, Gemma 7B, **ours** |
| **MQA:** Multi-Query (Shazeer, Google, 2019) | All heads share **one** K/V | PaLM, Gemma 2B |
| **GQA:** Grouped-Query (Google, 2023) | Groups of heads share K/V | Gemma 2/3, Llama 3 (8 KV heads), Qwen2.5 (0.5B: 14 query / 2 KV heads), gpt-oss (64 query / 8 KV heads) |
| **MLA:** Multi-head Latent Attention (DeepSeek, 2024) | Compress K/V into a small latent vector and expand it when needed | DeepSeek-V2/V3, Kimi K2. DeepSeek-V2 reported a **93% smaller KV cache** than its MHA predecessor |

**In our model:** `n_head = 6`, `d_head = 384 / 6 = 64`, plain MHA.

**QK-norm:** Gemma 3 and Qwen3 normalize Q and K before the dot product, for training stability.

---

## 7. Feed-forward network (MLP)

**Idea:** after attention mixes information *between* tokens, the MLP processes each token *on its own*. It's widely thought to be where much of the model's factual knowledge lives.

**Formulas:**
- **Classic** (GPT-2, ours): `FFN(x) = W₂ · GELU(W₁·x)`, with an inner size of `4 × d_model`
- **SwiGLU** (most modern models): `FFN(x) = W₂ · ( SiLU(W₁·x) ⊙ (W₃·x) )`, a *gated* MLP with 3 matrices
- **GeGLU** (Gemma): the same with GELU in place of SiLU

**In our model:** `384 → 1536 → 384` with GELU (ReLU in Karpathy's video; try both).

**In the frontier models:**

| Model | Activation |
|---|---|
| Original Transformer (Google) | ReLU |
| GPT-2 / GPT-3 (OpenAI) | GELU |
| gpt-oss (OpenAI) | SwiGLU (in MoE experts) |
| PaLM (Google) | SwiGLU |
| Gemma (Google) | GeGLU |
| Qwen2.5/3, DeepSeek-V3, Llama 3 | SwiGLU |

### Mixture of Experts (MoE)

**Idea:** replace the one big MLP with many smaller "expert" MLPs, plus a **router** that sends each token to only a few of them. The model gets the knowledge of a huge model at the compute cost of a small one.

**Formula:** `y = Σ_{i ∈ top-k} g_i(x) · Expert_i(x)`, where `g = softmax(router(x))` keeps only the top-k experts

| Model | Total parameters | Active per token | Experts |
|---|---|---|---|
| gpt-oss-120b (OpenAI) | ~117B | ~5.1B | 128, top-4 |
| gpt-oss-20b (OpenAI) | ~21B | ~3.6B | 32, top-4 |
| Qwen3-235B-A22B (Alibaba) | 235B | 22B | 128, top-8 |
| DeepSeek-V3 | 671B | 37B | 256 routed (top-8) + 1 shared |
| Kimi K2 (Moonshot) | ~1T | ~32B | 384 |

---

## 8. Residual connections — "the residual stream"

**Idea:** each sub-layer *adds* its output to its input instead of replacing it. Gradients flow straight through, so deep stacks train.

**Formula:** `x = x + Attention(Norm(x))`, then `x = x + FFN(Norm(x))`

**From Anthropic's research:** *A Mathematical Framework for Transformer Circuits* reframes this as the **residual stream**: a shared communication channel that every layer reads from and writes to. Attention heads and MLPs are components that read information, process it and write results back. This framing underlies much of modern interpretability research.

**In our model:** 6 blocks, each adding two updates to a 384-dimensional stream.

---

## 9. Normalization: LayerNorm → RMSNorm, post-norm → pre-norm

**Idea:** rescale activations to a stable range so training doesn't blow up.

**Formulas:**
- **LayerNorm:** `y = (x − mean) / √(var + ε) · γ + β`
- **RMSNorm:** `y = x / √(mean(x²) + ε) · γ`. No mean subtraction and no β, so it's cheaper with similar quality.
- **Post-norm** (original 2017 Transformer): `Norm(x + Sublayer(x))`
- **Pre-norm** (GPT-2 onward): `x + Sublayer(Norm(x))`. Much more stable for deep models.

**In our model:** pre-norm LayerNorm, as in Karpathy's video.

**In the frontier models:**

| Model | Norm | Placement |
|---|---|---|
| Original Transformer (Google) | LayerNorm | Post-norm |
| GPT-2 / GPT-3 (OpenAI) | LayerNorm | Pre-norm |
| gpt-oss (OpenAI) | RMSNorm | Pre-norm |
| Gemma 2/3 (Google) | RMSNorm | Pre- **and** post-norm |
| Qwen2.5/3, DeepSeek-V3, Llama 3 | RMSNorm | Pre-norm |

---

## 10. Output: logits, softmax, temperature, sampling

**Idea:** the final vector is projected to one score (**logit**) per vocabulary entry. Softmax turns the scores into probabilities, and we sample the next token from them.

**Formulas:**
- `logits = Norm(x) · W_outᵀ`, with shape `(vocab_size,)`
- `p_i = exp(z_i / τ) / Σ_j exp(z_j / τ)`, where **τ = temperature**: below 1 is safer and more repetitive; above 1 is more creative and more chaotic
- **Top-k:** keep only the k most likely tokens. **Top-p (nucleus):** keep the smallest set whose probabilities sum to p.

**In our model:** 72 logits per step. Try τ = 0.6, 0.8 and 1.2 and compare the poetry. Higher τ breaks the meter (*baḥr*) sooner.

**In the frontier models:** chat APIs from OpenAI, Google, Anthropic, Qwen and DeepSeek all expose `temperature`, and most expose `top_p` as well. **Logit soft-capping:** Gemma 2 squashes logits with `c · tanh(z / c)` to keep them bounded.

---

## 11. Loss: cross-entropy and perplexity

**Idea:** reward the model for putting high probability on the true next token.

**Formulas:**
- `loss = −(1/T) Σ_t log p(token_t | tokens_<t)`
- **Starting loss** for uniform guessing = `ln(vocab_size)`
- **Perplexity** = `exp(loss)`, roughly "how many choices the model is torn between"

**In our model:**
- Starting loss = ln(72) ≈ **4.28**. If your untrained model shows something very different, there's a bug.
- Shakespeare sanity check (Karpathy's config): validation loss ≈ 1.48 → perplexity ≈ 4.4
- Arabic poetry: unknown. **We'll be the ones to measure it.**

**In the frontier models:** the same loss, over trillions of tokens. DeepSeek-V3 adds **Multi-Token Prediction (MTP)**: extra heads also predict tokens t+2, …, a denser training signal that can also speed up generation.

---

## 12. Parameter counting

**Idea:** knowing where the parameters live tells you the memory, compute and speed (see the `ndim-infra` glossary: memory ≈ 16 bytes per parameter for training).

**Formula** (per block, ignoring biases and norms): `≈ 12 · d_model²`
- attention: `4 · d²` (W_q, W_k, W_v, W_o)
- MLP: `8 · d²` (d → 4d → d)

**In our model:** 12 × 384² × 6 blocks ≈ **10.6M**, plus embeddings ≈ **10.7M** total.

**In the frontier models:**

| Model | Layers | d_model | Heads | Parameters |
|---|---|---|---|---|
| Original Transformer, base (Google) | 6 + 6 | 512 | 8 | 65M |
| GPT-2 small (OpenAI) | 12 | 768 | 12 | 124M |
| GPT-2 XL (OpenAI) | 48 | 1,600 | 25 | 1.5B |
| GPT-3 (OpenAI) | 96 | 12,288 | 96 | 175B |
| Qwen2.5-0.5B (Alibaba) | 24 | 896 | 14 (2 KV) | 0.5B |
| DeepSeek-V3 | 61 | 7,168 | 128 | 671B total / 37B active |
| GPT-4/5, Claude, Gemini | Not published | | | |

---

## 13. Training: AdamW, learning rate, mixed precision

**Idea:** compute the loss, backpropagate gradients, nudge every parameter downhill. Repeat.

**Formulas:**
- **AdamW:** `m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)g²`, `θ ← θ − lr · ( m̂ / (√v̂ + ε) + λθ )`
- **Warmup + cosine decay:** ramp the learning rate up, then decay it smoothly (Module 2)
- **Training compute** ≈ `6 · N · D` FLOPs (N parameters, D tokens)

**In our model:** AdamW, lr = 3e-4, about 5,000 steps. Use **fp16 autocast** on the Colab T4 and **bf16** on the laptop (see the `ndim-infra` README for why).

**In the frontier models:** AdamW is the default nearly everywhere (GPT-3, Llama, Qwen). DeepSeek-V3 trained in **FP8** mixed precision at scale. Kimi K2 used the **Muon** optimizer (MuonClip) instead of AdamW.

---

## 14. Looking inside: interpretability (preview of Phase 4)

**From Anthropic's research:**
- **Superposition** (*Toy Models of Superposition*, 2022): models pack more concepts than they have neurons, so single neurons are messy.
- **Sparse autoencoders** (*Towards Monosemanticity*, 2023): decompose activations into clean, human-readable **features**.
- ***Scaling Monosemanticity*** (2024): found millions of features inside Claude 3 Sonnet. Turning up one "Golden Gate Bridge" feature produced **"Golden Gate Claude"**, a model that steered every conversation back to the bridge.

**Our stretch goal:** find a "rhyme letter" or "meter" feature inside Shaer-GPT.

---

## Cheat sheet: one row per model family

| | GPT-2 (OpenAI) | gpt-oss (OpenAI) | Gemma 3 (Google) | Claude (Anthropic) | Qwen3 (Alibaba) | DeepSeek-V3 | **Shaer-GPT (ours)** |
|---|---|---|---|---|---|---|---|
| Tokenizer | BPE 50k | BPE o200k | SentencePiece 262k | n/p | BPE ~151k | BPE ~128k | **Characters, 72** |
| Positions | Learned | RoPE | RoPE | n/p | RoPE | RoPE | **Learned** |
| Attention | MHA | GQA + sliding window | GQA, 5 local:1 global | n/p | GQA + QK-norm | MLA | **MHA** |
| MLP | GELU | MoE SwiGLU | GeGLU | n/p | SwiGLU (+MoE) | MoE SwiGLU | **GELU** |
| Norm | LN pre | RMS pre | RMS pre+post | n/p | RMS pre | RMS pre | **LN pre** |
| Context | 1k | 128k | 128k | 200k | 32k–128k | 128k | **256 chars** |

*n/p = not published*

---

## Sources

**Foundations:** [Attention Is All You Need](https://arxiv.org/html/1706.03762v7) (Google, 2017) · Karpathy, [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)

**OpenAI:** [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) · [GPT-3](https://arxiv.org/abs/2005.14165) · [gpt-oss](https://openai.com/index/introducing-gpt-oss/) · [tiktoken](https://github.com/openai/tiktoken)

**Google:** [MQA](https://arxiv.org/abs/1911.02150) · [PaLM](https://arxiv.org/abs/2204.02311) · [GQA](https://arxiv.org/abs/2305.13245) · [Gemma](https://arxiv.org/abs/2403.08295) · [Gemma 2](https://arxiv.org/abs/2408.00118) · [Gemma 3](https://arxiv.org/abs/2503.19786)

**Anthropic:** [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) · [Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) · [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) · [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) · [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

**Chinese labs:** [Qwen2.5](https://arxiv.org/abs/2412.15115) · [Qwen3](https://arxiv.org/abs/2505.09388) · [DeepSeek-V2 (MLA)](https://arxiv.org/abs/2405.04434) · [DeepSeek-V3](https://arxiv.org/abs/2412.19437) · [Kimi K2](https://github.com/MoonshotAI/Kimi-K2)

**Other building blocks:** [RoPE](https://arxiv.org/abs/2104.09864) · [SwiGLU](https://arxiv.org/abs/2002.05202) · [RMSNorm](https://arxiv.org/abs/1910.07467) · [Llama 3](https://arxiv.org/abs/2407.21783)

**Data:** [arbml/ashaar](https://huggingface.co/datasets/arbml/ashaar), ~254k Arabic poems from aldiwan.net
