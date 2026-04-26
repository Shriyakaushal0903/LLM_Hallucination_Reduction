# LLM Hallucination Reduction

A research project exploring fine-tuning techniques to reduce hallucination in large language models using **Dirichlet Evidential Learning** and **Orthogonal Fine-Tuning (OFT)**.

---

## Overview

This project implements a two-stage fine-tuning pipeline on top of LLaMA 3.1 (8B) to reduce hallucination behavior. The core idea combines two complementary approaches:

1. **Dirichlet Loss (Evidential Deep Learning)** — replaces the standard cross-entropy loss with an uncertainty-aware loss function that models the output distribution as a Dirichlet distribution. This encourages the model to express lower confidence when evidence is weak, rather than confidently generating incorrect information.

2. **Orthogonal Fine-Tuning (OFT / COFT)** — applies orthogonal constraints to attention weight updates, preserving the hyperspherical structure of pretrained representations and preventing "mode collapse" or repetitive generation loops that are often associated with hallucination.

Evaluation is performed using the [HalluLens](https://github.com/amazon-science/hallulens) benchmark suite across three tasks.

---

## Results

Model: **Dirichlet + Ortho-FT Fine-Tuned (7B)**

| Task | Metric | Score |
|---|---|---|
| **PreciseWikiQA** | False Refusal Rate | 67.19% |
| | Hallucination Rate | 42.68% |
| | Correct Answer Rate | 14.29% |
| **LongWiki** | False Refusal Rate | 19.93% |
| | Recall@32 | 65.21% |
| | Precision | 53.43% |
| | F1@32 | 58.45% |
| **Nonexistent Entities** | Mixed Entities | 19.25% |
| | Generated Entities | 4.05% |
| | False Acceptance Rate | 11.65% |


---

## Project Structure

```
LLM_Hallucination_Reduction/
├── fine_tune.ipynb           # Main training notebook (Dirichlet + OFT pipeline)
├── prepare_and_test.sh       # Quick inference test using a local checkpoint
├── run_hallulens_local.sh    # Runs HalluLens evaluation against a local model
└── results.json              # Benchmark evaluation results
```

---

## Method Details

### Stage 1 — Dirichlet Evidential Fine-Tuning

A custom `EvidentialTrainer` (subclassing `SFTTrainer`) overrides the loss computation:

- Raw logits are passed through `softplus` to produce non-negative **evidence** values.
- Evidence is converted to **Dirichlet parameters** `α = evidence + 1`.
- The **expected probability** `E[p] = α / S` is used to compute NLL loss.

This replaces the standard softmax + cross-entropy loss, making the model explicitly reason about total evidence before committing to a token prediction.

### Stage 2 — Orthogonal Fine-Tuning

A second training round applies OFT adapters (`coft=True`) to attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`). A custom `EvidentialOrthogonalTrainer` adds an **orthogonality regularization term** to the loss:

```
total_loss = nll_loss + λ * ortho_loss
```

where `ortho_loss` penalizes off-diagonal cosine similarity between hidden state token representations (Frobenius norm of `similarity - identity`). This discourages repetitive or looping generation patterns.

### Base Model & Training Setup

- **Base model:** `unsloth/Meta-Llama-3.1-8B` (4-bit quantized via bitsandbytes)
- **PEFT method:** LoRA (`r=16`, applied to all linear layers) for Stage 1; OFT (`r=8`) for Stage 2
- **Optimizer:** AdamW 8-bit
- **Precision:** bfloat16
- **Training steps:** 60 steps per stage

---

