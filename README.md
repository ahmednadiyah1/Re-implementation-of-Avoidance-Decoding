# Avoidance Decoding (Unofficial Implementation)

An unofficial, in-progress PyTorch implementation of the decoding algorithm from:

> Kyeongman Park, Nakyeong Yang, and Kyomin Jung. **Avoidance Decoding for Diverse
> Multi-Branch Story Generation.** EMNLP 2025.
> [arXiv:2509.02170](https://arxiv.org/abs/2509.02170) ·
> [ACL Anthology](https://aclanthology.org/2025.emnlp-main.381/)

This repo implements the paper's **Algorithm 1**: a training-free decoding strategy
that penalizes a candidate token's similarity to previously generated stories
("negative samples"), so that multiple stories sampled from the same prompt stay
diverse instead of converging on the same characters, setting, and plot.

This is **not affiliated with the paper's authors**, and it is **not a full
replication** of the paper (no reproduction of its reported BLEU/ROUGE-L/METEOR/
Sent-Sim/LLMScore/Degen numbers, no baselines, no human eval) — it implements the
core decoding mechanism only.

## How it works

At each decoding step, the top-`k` candidate next-tokens are scored not just by
the model's own logits, but by how similar they'd make the story to the negative
samples, via two penalties that shift weight over the course of generation:

- **Concept-level Similarity Penalty (CSP)** — max cosine similarity between a
  candidate token's last hidden state and every individual token's hidden state
  across all negative samples. Dominant early in generation, to diversify
  premise/plot/characters.
- **Narrative-level Similarity Penalty (NSP)** — cosine similarity between a
  sentence embedding of the story-so-far (with the candidate appended) and each
  negative sample's sentence embedding. Weighted more heavily later in generation,
  to keep the plot diverse without breaking coherence.
- **Hybrid penalty**: `γ = δ + (1-δ)·sigmoid(t - T0)`, mixing CSP and NSP as
  `γ·CSP + (1-γ)·NSP`, then taking the max over negative samples (an L∞-style
  penalty per the paper's Eq. 4) and scaling by `β`.
- **Adaptive `k`/`α`** — the candidate pool size `k` and penalty weight `α` are
  recomputed at every step from the KL divergence of the model's current logits
  against a uniform distribution and against a near-one-hot "certain" distribution
  (following the Adaptive Contrastive Search idea the paper builds on).
- Final candidate score: `F = (1-α)·p(token) - α·similarity_penalty`; the token
  with the highest `F` is selected greedily.

## Repository layout

| File | Purpose |
|---|---|
| `avoidance_decoding.py` | CLI entry point: loads a Hugging Face causal LM, obtains negative samples, and runs avoidance decoding for each prompt. |
| `negative_sampling.py` | `NegativeSamples` — generates negative story samples per prompt (via plain temperature sampling) and caches their hidden states / sentence embeddings. |
| `model.py` | `AvoidanceDecodingModel` — computes adaptive `k`, `α`, `γ` and drives the per-step decoding loop. |
| `utils.py` | Core algorithm: top-`k` candidate expansion, the hybrid CSP + NSP penalty, ranking, and the single-step `avoidance_decoding` function. |
| `prompts.txt` | Sample story-writing prompts used as decoding input. |
| `loss_fun.py` | Placeholder — currently empty; this is decoding-time only, no training objective. |
| `results/negative_samples.json` | Example cached negative samples from a prior run. |

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `transformers`, `accelerate`, `peft`, and
`sentence-transformers`. Narrative-level (sentence) embeddings are computed with
`Qwen/Qwen3-Embedding-0.6B` via `sentence-transformers`, in place of the paper's
Sentence-BERT — this will be downloaded automatically on first run.

## Intended usage

Generate negative samples from a prompt file, then run avoidance decoding:

```bash
python avoidance_decoding.py \
  --model <hf-model-name-or-path> \
  --data prompts.txt \
  --output_negative_samples_file ./results \
  --output_dir ./results/generations.json \
  --beta 2 \
  --delta 0.5 \
  --T0 25
```

Or reuse previously generated negative samples instead of regenerating them:

```bash
python avoidance_decoding.py \
  --model <hf-model-name-or-path> \
  --negative_samples_file ./results/negative_samples.json \
  --output_dir ./results/generations.json
```

| Flag | Meaning |
|---|---|
| `--model` | HF model name or local path (required). |
| `--data` | Path to a prompts file (one prompt per line) to generate negative samples from. |
| `--output_negative_samples_file` | Directory to save generated negative samples to. |
| `--negative_samples_file` | Path to a previously saved `negative_samples.json` to load instead of regenerating. |
| `--output_dir` | Where to write generated story outputs (required). |
| `--decoding_len` | Number of tokens to generate per branch (default `50`). |
| `--beta` | Scaling factor applied to the hybrid similarity penalty (default `2`). |
| `--delta` | Lower bound for `γ`'s CSP↔NSP schedule (default `0.5`). |
| `--T0` | Decoding-step midpoint of the `γ` sigmoid schedule (default `25`). |
