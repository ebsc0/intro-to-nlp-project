# PROJECT.md: Multilingual Character Prediction for Space Communications

## CS498 Introduction to NLP - Course Project

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Architecture Overview](#3-architecture-overview)
4. [CANINE Encoder](#4-canine-encoder)
5. [LoRA Fine-tuning](#5-lora-fine-tuning)
6. [Character Prediction Head](#6-character-prediction-head)
7. [Why Causal Masking Is Not Required](#7-why-causal-masking-is-not-required)
8. [Data Pipeline](#8-data-pipeline)
9. [Training Configuration](#9-training-configuration)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Constraints Compliance](#11-constraints-compliance)
12. [Evaluation Strategy](#12-evaluation-strategy)
13. [Implementation Timeline](#13-implementation-timeline)
14. [Risk Mitigation](#14-risk-mitigation)
15. [Alternatives Considered](#15-alternatives-considered)

---

## 1. Executive Summary

This project implements an intelligent auto-completion system for astronaut-mission control communications. The system predicts the next character given a context string, supporting 10 languages across multiple writing systems.

### Core Architecture

```
CANINE Encoder (pretrained, 130M params)
    + LoRA Adapters (trainable, ~1.2M params)
    + Standard Padding Attention
    + Character Prediction Head (trainable, ~4M params)
    ─────────────────────────────────────────────
    = Total: ~135M params, ~5.2M trainable
    = Checkpoint size: ~10-15 MB (LoRA + Head only)
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| CANINE encoder | Only pretrained character-level multilingual model available |
| LoRA fine-tuning | Parameter-efficient, preserves pretrained knowledge |
| Padding-only prefix modeling | Inputs already exclude the target character, so no future-target leakage exists |
| Character vocabulary | Direct character prediction, no tokenization artifacts |

---

## 2. Problem Statement

### Task Definition

Given a context string of characters, predict the most likely next character. The system outputs 3 guesses, and evaluation is case-insensitive (correct if the gold answer appears in any of the 3 guesses).

### Input/Output Format

```
Input:  "The astronaut Alan Shepard h"
Output: "ase"  (guessing 'a', 's', 'e' - correct if next char is 'a' for "has")
```

### Language Coverage

The system must handle 10 languages across 6 writing systems:

| Language | Script | Example |
|----------|--------|---------|
| English | Latin | "The astronaut..." |
| French | Latin | "L'astronaute..." |
| German | Latin | "Der Astronaut..." |
| Italian | Latin | "L'astronauta..." |
| Russian | Cyrillic | "Астронавт..." |
| Chinese | Han/CJK | "宇航员..." |
| Japanese | Han + Kana | "宇宙飛行士..." |
| Korean | Hangul | "우주비행사..." |
| Hindi | Devanagari | "अंतरिक्ष यात्री..." |
| Arabic | Arabic | "رائد الفضاء..." |

### Domain

Space mission communications, specifically:
- Astronaut-ground dialogue
- Mission status reports
- Recovery operations
- Technical callouts

---

## 3. Architecture Overview

### High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: "Der Astronaut Alan Shepard ist gerade aus der Kapsel gestiegen"  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      CHARACTER TOKENIZATION                          │  │
│  │  Text → Unicode codepoints → Input IDs                               │  │
│  │  "Der " → [68, 101, 114, 32, ...]                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                       CANINE ENCODER                                 │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Character Embeddings (hash-based, 768d)                        │ │  │
│  │  │ ├── Handles any Unicode character (no OOV)                     │ │  │
│  │  │ └── Pretrained on 104 languages                                │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Local Transformer Block (1 layer)                              │ │  │
│  │  │ └── Captures local character patterns                          │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Strided Convolution (4x downsampling)                          │ │  │
│  │  │ └── Reduces sequence length for efficiency                     │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Deep Transformer (12 layers, 768d, 12 heads)                   │ │  │
│  │  │ ├── FROZEN pretrained weights (W)                              │ │  │
│  │  │ ├── TRAINABLE LoRA adapters (A, B matrices)                    │ │  │
│  │  │ ├── CAUSAL ATTENTION MASK applied                              │ │  │
│  │  │ └── Global context modeling                                    │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Upsampling (back to character resolution)                      │ │  │
│  │  │ └── Restores original sequence length                          │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                      │  │
│  │  Output: Hidden states [batch, seq_len, 768]                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼ (extract last position)               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                   CHARACTER PREDICTION HEAD                          │  │
│  │                                                                      │  │
│  │  Input: last_hidden_state [batch, 768]                              │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Linear(768 → 512)                                              │ │  │
│  │  │ GELU activation                                                 │ │  │
│  │  │ Dropout(0.1)                                                    │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌────────────────────────────────────────────────────────────────┐ │  │
│  │  │ Linear(512 → vocab_size)  # bounded learned output vocabulary  │ │  │
│  │  └────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  Mask <PAD>/<UNK> → Top-3 logit selection                           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  OUTPUT: top-3 vocabulary IDs (converted to characters outside model)      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Parameters | Trainable | Purpose |
|-----------|------------|-----------|---------|
| CANINE Embeddings | ~2M | No | Character representations |
| CANINE Transformer | ~128M | No (base) | Contextual encoding |
| LoRA Adapters | ~1.2M | Yes | Domain adaptation |
| Prediction Head | ~4M | Yes | Character classification |
| **Total** | ~135M | ~5.2M | |

---

## 4. CANINE Encoder

### What is CANINE?

CANINE (Character Architecture with No tokenization In Neural Encoders) is a character-level transformer model developed by Google Research. Unlike traditional models that use subword tokenization (BPE, WordPiece), CANINE operates directly on Unicode characters.

### Model Specifications

| Attribute | Value |
|-----------|-------|
| Model ID | `google/canine-c` |
| Parameters | 130M |
| Hidden dimension | 768 |
| Attention heads | 12 |
| Transformer layers | 12 (deep) + 1 (local) |
| Max sequence length | 2048 characters |
| Pretraining data | Wikipedia (104 languages) |
| Pretraining objective | Character-level pretraining objective |

### Why CANINE for This Task?

#### Advantage 1: Native Character-Level Processing

```
Traditional Subword Model (e.g., BERT):
  "astronaut" → ["astro", "##naut"] → [15432, 8876]
  Problem: Token boundaries don't align with character boundaries

CANINE:
  "astronaut" → ['a','s','t','r','o','n','a','u','t'] → [97,115,116,114,111,110,97,117,116]
  Advantage: Direct character-to-character mapping
```

#### Advantage 2: Unified Multilingual Representation

```
All languages use the same representation space:

English:  "astronaut" → character IDs → [768d] hidden states
Russian:  "астронавт" → character IDs → [768d] hidden states
Chinese:  "宇航员"     → character IDs → [768d] hidden states

No language-specific tokenizers needed!
```

#### Advantage 3: No Out-of-Vocabulary (OOV) Problem

CANINE uses hash-based embeddings that can represent any Unicode character:

```python
# Traditional embedding (can have OOV)
embedding = nn.Embedding(vocab_size, hidden_dim)  # Fixed vocab

# CANINE hash-based embedding (no OOV)
def canine_embed(char):
    codepoint = ord(char)  # Unicode codepoint
    hash_bucket = hash(codepoint) % num_buckets
    return embedding_table[hash_bucket]  # Always has a representation
```

### CANINE Internal Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CANINE ENCODER DETAILS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Character sequence [N characters]                                   │
│                                                                             │
│  1. CHARACTER EMBEDDING                                                     │
│     ├── Hash-based embedding (handles any Unicode)                         │
│     ├── Dimension: 768                                                      │
│     └── Output: [N, 768]                                                   │
│                                                                             │
│  2. LOCAL TRANSFORMER (1 layer)                                            │
│     ├── Window size: 128 characters                                        │
│     ├── Captures: n-gram like patterns                                     │
│     └── Output: [N, 768]                                                   │
│                                                                             │
│  3. STRIDED CONVOLUTION                                                     │
│     ├── Stride: 4 (downsample by 4x)                                       │
│     ├── Purpose: Reduce sequence length for efficiency                     │
│     └── Output: [N/4, 768]                                                 │
│                                                                             │
│  4. DEEP TRANSFORMER (12 layers)                                           │
│     ├── Full self-attention                                                 │
│     ├── 12 attention heads                                                 │
│     ├── FFN hidden dim: 3072                                               │
│     ├── This is where LoRA adapters are injected                          │
│     └── Output: [N/4, 768]                                                 │
│                                                                             │
│  5. UPSAMPLING                                                              │
│     ├── Restores original resolution                                       │
│     ├── Uses learned upsampling convolution                                │
│     └── Output: [N, 768]                                                   │
│                                                                             │
│  Final Output: Hidden states for each character position                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. LoRA Fine-tuning

### What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that:
1. Freezes pretrained model weights
2. Injects small trainable matrices into each layer
3. Trains only these small matrices (~1% of total parameters)

### Mathematical Foundation

For a pretrained weight matrix W ∈ R^(d×d):

```
Standard fine-tuning:
  W' = W + ΔW           where ΔW ∈ R^(d×d)
  Trainable params: d² = 768² = 589,824 per matrix

LoRA fine-tuning:
  W' = W + (α/r)·(B·A)  where A ∈ R^(d×r), B ∈ R^(r×d)
  Trainable params: 2·d·r = 2·768·16 = 24,576 per matrix
  Reduction: 96% fewer parameters
```

### Visual Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LoRA MECHANISM                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                     FROZEN                      TRAINABLE                   │
│                       │                            │                        │
│                       ▼                            ▼                        │
│               ┌──────────────┐            ┌──────────────┐                 │
│               │              │            │              │                 │
│     Input     │      W       │            │   B    A     │                 │
│    [768] ────►│  [768×768]   │───────────►│[768×r][r×768]│                 │
│               │   (frozen)   │     +      │ (trainable)  │                 │
│               │              │            │   r = 16     │                 │
│               └──────────────┘            └──────────────┘                 │
│                       │                            │                        │
│                       └──────────┬─────────────────┘                       │
│                                  │                                          │
│                                  ▼                                          │
│                             Output [768]                                    │
│                          = W·x + (α/r)·(B·A·x)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LoRA Configuration

```python
LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=16,                    # Rank of low-rank matrices
    lora_alpha=32,           # Scaling factor (effective scale = alpha/r = 2)
    target_modules=[         # Which layers to adapt
        "query",             # Attention Q projection
        "key",               # Attention K projection
        "value",             # Attention V projection
        "attention.output.dense",  # Attention output projection
    ],
    lora_dropout=0.1,        # Regularization
    bias="none",             # Don't train bias terms
)
```

`TaskType.FEATURE_EXTRACTION` is the correct PEFT setting here because the implementation uses `CanineModel` as an encoder and adds a custom next-character classification head. It is not using a `CausalLM` model with a built-in language-modeling head.

### Which Layers Get LoRA?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CANINE TRANSFORMER LAYER (× 12)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Self-Attention Block:                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  W_q [768 × 768]  ← LoRA applied (A_q, B_q)                       │    │
│  │  W_k [768 × 768]  ← LoRA applied (A_k, B_k)                       │    │
│  │  W_v [768 × 768]  ← LoRA applied (A_v, B_v)                       │    │
│  │  W_o [768 × 768]  ← LoRA applied (A_o, B_o)                       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Feed-Forward Block:                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  W_up   [768 × 3072]   (frozen, no LoRA)                          │    │
│  │  W_down [3072 × 768]   (frozen, no LoRA)                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Per-layer LoRA params: 4 × 2 × 768 × 16 = 98,304                          │
│  Total LoRA params (12 layers): 12 × 98,304 = 1,179,648 ≈ 1.2M            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Comparison

| Configuration | Encoder Params | Head Params | Total Trainable | % of Model |
|--------------|----------------|-------------|-----------------|------------|
| Head only | 0 | 4M | 4M | 3.0% |
| LoRA + Head | 1.2M | 4M | 5.2M | 3.9% |
| Full fine-tune | 130M | 4M | 134M | 100% |

### Benefits of LoRA for This Project

| Benefit | Explanation |
|---------|-------------|
| **Preserves pretraining** | Frozen weights retain 104-language knowledge |
| **Prevents overfitting** | Low-rank constraint acts as regularizer |
| **Fast training** | 96% fewer gradients to compute |
| **Small checkpoint** | Only save LoRA matrices + head (~10 MB) |
| **Memory efficient** | Fewer optimizer states needed |

---

## 6. Character Prediction Head

### Purpose

The prediction head converts CANINE's hidden representations into character probabilities. This component is **trained from scratch** because:

1. CANINE was trained for MLM (masked token prediction), not next-token prediction
2. CANINE outputs [768d] vectors, not character probabilities
3. We need to map to our specific character vocabulary

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CHARACTER PREDICTION HEAD                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: last_hidden_state [batch_size, 768]                                │
│         (hidden state at the final character position)                     │
│                                                                             │
│  Layer 1: Projection                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Linear(in_features=768, out_features=512)                         │    │
│  │  Parameters: 768 × 512 + 512 = 393,728                            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Layer 2: Activation                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  GELU (Gaussian Error Linear Unit)                                 │    │
│  │  GELU(x) = x · Φ(x) where Φ is standard normal CDF               │    │
│  │  Parameters: 0                                                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Layer 3: Regularization                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Dropout(p=0.1)                                                     │    │
│  │  Randomly zeros 10% of activations during training                 │    │
│  │  Parameters: 0                                                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Layer 4: Classification                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Linear(in_features=512, out_features=vocab_size)                  │    │
│  │  vocab_size ≈ 8,000 characters                                     │    │
│  │  Parameters: 512 × 8000 + 8000 = 4,104,000                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  Output: logits [batch_size, vocab_size]                                   │
│                                                                             │
│  Total Head Parameters: 393,728 + 4,104,000 = 4,497,728 ≈ 4.5M            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Character Vocabulary

The implementation uses two different representations:

- CANINE inputs use raw Unicode codepoints, so the encoder can consume any Unicode text without a tokenizer.
- Model outputs use a bounded learned character vocabulary built from the training targets.

The output vocabulary in `src/model.py` is created with `CharacterVocab.build(...)`:

```python
CharacterVocab.build(
    texts=targets,
    min_freq=1,
    max_size=8000,
)
```

That means the output side is:

- data-built rather than hand-specified,
- capped by `max_size`,
- frequency-ordered,
- and initialized with explicit special tokens.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHARACTER VOCABULARY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Special Tokens:                                                            │
│  ├── <PAD>: 0    (reserved padding label)                                  │
│  └── <UNK>: 1    (reserved fallback label)                                 │
│                                                                             │
│  Remaining IDs:                                                             │
│  ├── Added from observed target characters                                 │
│  ├── Ordered by frequency in the training targets                          │
│  ├── Limited to max_size - 2 entries                                       │
│  └── Saved as char_to_id JSON and reconstructed as id_to_char at load time │
│                                                                             │
│  Important distinction:                                                     │
│  ├── Input side: raw Unicode codepoints for CANINE                         │
│  └── Output side: bounded learned character vocabulary                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Top-3 Selection

```python
def predict_topk(logits, k=3, forbidden_ids=(PAD_ID, UNK_ID)):
    """
    Return top-k vocabulary indices from model logits.
    
    Args:
        logits: [batch_size, vocab_size] raw model outputs
        k: number of predictions to return
        forbidden_ids: output IDs to exclude from ranking
    
    Returns:
        Tensor of shape [batch_size, k] containing predicted vocab IDs
    """
    masked_logits = logits.clone()
    vocab_size = masked_logits.size(-1)

    for idx in forbidden_ids:
        if 0 <= idx < vocab_size:
            masked_logits[:, idx] = float("-inf")

    safe_k = min(k, vocab_size)
    return torch.topk(masked_logits, k=safe_k, dim=-1).indices
```

`CanineLoRACharPredictor.predict_topk(...)` returns vocabulary indices only. The outer inference pipeline loads `vocab.json`, maps IDs back to characters, and formats the final three-character prediction strings.

---

## 7. Why Causal Masking Is Not Required

### Prefix-Only Supervision

This project trains on **prefix-only next-character examples**. Each training input contains only the observed context, and the gold next character is stored separately as the label.

```
Input context:  "The astronaut Alan Shepard h"
Target label:   "a"
```

Because the target character is never included in the input sequence, the model does not have future-target information to mask away.

### Attention Behavior Used in This Project

The current implementation uses CANINE's standard bidirectional encoder attention together with a standard 2D padding mask:

- Real characters can attend to other real characters in the observed prefix.
- Padding positions are masked out.
- The model predicts from the hidden state at the **last non-padding position** of the observed context.

This is sufficient for the supervised setup used here because:

1. Each example contains only the observed prefix.
2. The target next character is excluded from the input.
3. The prediction head reads only the representation at the final observed position.
4. Therefore, there is no future-target leakage to prevent with a triangular causal mask.

### Why Padding-Only Attention Is Acceptable

Standard 2D padding masks are simpler and align with the implemented model:

- They prevent attention to pad tokens.
- They preserve CANINE's pretrained bidirectional prefix representations.
- They avoid additional masking logic inside CANINE's multi-stage encoder.

In this project, causal masking would only be necessary if future characters were present in the input sequence, or if the model were being redesigned to behave as a strict autoregressive decoder rather than a prefix-conditioned classifier.

### Performance Note

For this task, bidirectional prefix encoding is acceptable and may be preferable to causal masking. The last observed position already has access only to the provided prefix, so adding triangular masks increases implementation complexity without addressing a real label-leakage problem in the current data formulation.

---

## 8. Data Pipeline

### Data Sources

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRIMARY: NASA Mission Transcripts                                         │
│  ├── Mercury Program (matches sample data)                                 │
│  │   └── Alan Shepard's Freedom 7 mission                                 │
│  ├── Apollo Program                                                        │
│  │   ├── Apollo 11 (Moon landing)                                         │
│  │   ├── Apollo 13 (Famous "Houston, we have a problem")                  │
│  │   └── Other Apollo missions                                            │
│  ├── Space Shuttle Program                                                 │
│  │   └── Various mission transcripts                                      │
│  └── ISS Communications                                                    │
│      └── Astronaut-ground dialogue                                        │
│                                                                             │
│  Source: NASA Technical Reports Server (NTRS)                              │
│  Format: Text transcripts                                                   │
│  Language: English (original)                                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TRANSLATION: Parallel Corpus Generation                                   │
│  ├── Method: Google Translate API / DeepL API                             │
│  ├── Target Languages: RU, ZH, JA, HI, AR, KO, FR, DE, IT                 │
│  └── Quality: Machine translation (acceptable for training)               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SUPPLEMENTARY: Domain-Relevant Text                                       │
│  ├── Wikipedia articles                                                    │
│  │   ├── Space exploration                                                │
│  │   ├── Astronaut biographies                                            │
│  │   ├── Space missions                                                   │
│  │   └── Spacecraft descriptions                                          │
│  ├── ESA/JAXA/Roscosmos public communications                            │
│  └── Space news articles                                                   │
│                                                                             │
│  Estimated Total: 50-100 MB of text across 10 languages                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA PROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 1: Collection                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Raw transcripts (English)                                           │  │
│  │  "Hello...this is the aircraft carrier Lake Champlain..."           │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  Step 2: Cleaning                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  - Normalize whitespace                                               │  │
│  │  - Remove control characters                                         │  │
│  │  - Standardize punctuation                                           │  │
│  │  - Split into sentences/utterances                                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  Step 3: Translation                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  For each sentence in English:                                       │  │
│  │    - Translate to Russian                                            │  │
│  │    - Translate to Chinese                                            │  │
│  │    - Translate to Japanese                                           │  │
│  │    - ... (all 9 target languages)                                   │  │
│  │  Result: 10x parallel corpus                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  Step 4: Vocabulary Building                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  - Count character frequencies across all texts                      │  │
│  │  - Include all characters with freq >= 2                             │  │
│  │  - Add script-specific character sets (ensure coverage)             │  │
│  │  - Create char_to_id and id_to_char mappings                        │  │
│  │  - Save vocabulary file                                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  Step 5: Training Example Generation                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  For each text:                                                       │  │
│  │    For i in range(1, len(text)):                                     │  │
│  │      context = text[:i]        # Characters 0 to i-1                 │  │
│  │      target = text[i]          # Character at position i             │  │
│  │      examples.append((context, target))                              │  │
│  │                                                                       │  │
│  │  Example:                                                             │  │
│  │  Text: "Hello"                                                        │  │
│  │  Examples: ("H", "e"), ("He", "l"), ("Hel", "l"), ("Hell", "o")      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                              │                                              │
│                              ▼                                              │
│  Step 6: Dataset Splits                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  - Training: 80%                                                      │  │
│  │  - Validation: 10%                                                    │  │
│  │  - Test: 10% (internal testing, separate from course eval)          │  │
│  │  - Ensure language balance in all splits                             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Format

```python
# Training example format
{
    'input_ids': [68, 101, 114, 32, 65, ...],  # Context characters as IDs
    'attention_mask': [1, 1, 1, 1, 1, ...],    # 1 for real tokens
    'label': 115,                              # Target character ID
    'language': 'de',                          # Language code (for analysis)
}

# Batch format
{
    'input_ids': tensor([batch_size, max_seq_len]),       # Padded
    'attention_mask': tensor([batch_size, max_seq_len]),  # 0 for padding
    'labels': tensor([batch_size]),                       # Target char IDs
}
```

---

## 9. Training Configuration

### Hyperparameters

```python
training_config = {
    # Model
    'base_model': 'google/canine-c',
    'vocab_size': 8000,  # upper bound for the learned output vocabulary
    
    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'lora_target_modules': ['query', 'key', 'value', 'attention.output.dense'],
    
    # Training
    'batch_size': 32,
    'max_seq_len': 512,
    'num_epochs': 5,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    
    # Learning rates
    'lr_encoder_lora': 1e-4,    # LoRA adapters
    'lr_head': 1e-3,            # Prediction head (higher, training from scratch)
    
    # Optimizer
    'optimizer': 'AdamW',
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    
    # Scheduler
    'scheduler': 'linear_warmup_decay',
    
    # Regularization
    'dropout': 0.1,
    'label_smoothing': 0.0,
    
    # Hardware
    'fp16': True,               # Mixed precision
    'gradient_checkpointing': False,  # Not needed (model is small enough)
}
```

### Training Loop

```python
# Pseudocode for training loop

model = CanineLoRACharPredictor(
    ModelConfig(
        base_model='google/canine-c',
        vocab_size=8000,
    )
)

# Separate parameter groups
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n], 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3},
], weight_decay=0.01)

scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=500,
    num_training_steps=total_steps
)

criterion = CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    
    for batch in train_dataloader:
        # Forward pass
        logits = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # Compute loss
        loss = criterion(logits, batch['labels'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
    
    # Validation
    model.eval()
    val_accuracy = evaluate(model, val_dataloader)
    print(f"Epoch {epoch}: Val Accuracy = {val_accuracy:.4f}")

# Save checkpoint
model.save('work/')
```

### Expected Training Metrics

| Metric | Expected Value |
|--------|----------------|
| Training loss (final) | 1.5 - 2.5 |
| Validation accuracy | 70-80% (top-3) |
| Training time | 2-4 hours (single GPU) |
| GPU memory | ~6 GB (with fp16) |

---

## 10. Inference Pipeline

### Loading the Model

```python
def load_model(work_dir):
    """Load trained model from checkpoint."""

    model = CanineLoRACharPredictor.load(work_dir, map_location="cpu")
    vocab = CharacterVocab.load(os.path.join(work_dir, "vocab.json"))

    return model, vocab
```

Checkpoint contract for the current implementation:

- `model_config.json`: serialized `ModelConfig`
- `canine_lora_adapters/`: PEFT LoRA adapter weights
- `char_head.pt`: character-classification head weights
- `vocab.json`: loaded by the outer inference pipeline, not by `CanineLoRACharPredictor.load()`

### Inference Function

```python
def predict(model, vocab, contexts, device='cuda'):
    """
    Generate top-3 predictions for each context.
    
    Args:
        model: Trained CanineLoRACharPredictor
        vocab: CharacterVocab
        contexts: List of context strings
        device: 'cuda' or 'cpu'
    
    Returns:
        List of prediction strings (e.g., ["aes", "tion", ...])
    """
    model.eval()
    model.to(device)
    
    predictions = []
    
    # Process in batches for efficiency
    batch_size = 32
    for i in range(0, len(contexts), batch_size):
        batch_contexts = contexts[i:i+batch_size]
        
        # Encode contexts as Unicode codepoints for CANINE
        input_ids = []
        for ctx in batch_contexts:
            ids = [ord(c) for c in ctx]
            if not ids:
                ids = [0]
            input_ids.append(ids)
        
        # Pad
        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        attention_masks = []
        for ids in input_ids:
            padding = [0] * (max_len - len(ids))
            padded_ids.append(ids + padding)
            attention_masks.append([1] * len(ids) + [0] * len(padding))
        
        # Convert to tensors
        input_ids_tensor = torch.tensor(padded_ids, device=device)
        attention_mask_tensor = torch.tensor(attention_masks, device=device)
        
        # Predict
        with torch.no_grad():
            top3 = model.predict_topk(input_ids_tensor, attention_mask_tensor, k=3)
        
        # Convert output vocabulary IDs back to characters
        for j in range(top3.shape[0]):
            chars = [vocab.get_char(idx.item()) for idx in top3[j]]
            predictions.append(''.join(chars))
    
    return predictions
```

### Inference Script (predict.sh)

```bash
#!/bin/bash
# src/predict.sh
# Usage: bash src/predict.sh <input_file> <output_file>

INPUT_FILE=$1
OUTPUT_FILE=$2

python src/myprogram.py test \
    --work_dir /job/work \
    --test_data "$INPUT_FILE" \
    --test_output "$OUTPUT_FILE"
```

### Batch Processing Optimization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INFERENCE OPTIMIZATION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Optimization 1: Batch Processing                                          │
│  ├── Process multiple contexts in parallel                                 │
│  ├── Batch size: 32-64 (depending on GPU memory)                          │
│  └── ~32x speedup vs sequential processing                                 │
│                                                                             │
│  Optimization 2: FP16 Inference                                            │
│  ├── Use half-precision floats                                            │
│  ├── 2x faster computation                                                 │
│  └── 2x less memory usage                                                  │
│                                                                             │
│  Optimization 3: LoRA Merge (Optional)                                     │
│  ├── Merge LoRA weights into base model before inference                  │
│  ├── W_merged = W + (alpha/r) * B @ A                                     │
│  └── Eliminates LoRA computation overhead                                 │
│                                                                             │
│  Optimization 4: Efficient Data Loading                                    │
│  ├── Read all inputs at once                                              │
│  ├── Pre-tokenize before batching                                         │
│  └── Use DataLoader with num_workers > 0                                  │
│                                                                             │
│  Expected Performance:                                                      │
│  ├── Input size: ~2000 examples (based on open-dev)                       │
│  ├── Time per batch (32 examples): ~0.1s                                  │
│  ├── Total inference time: ~10 seconds                                    │
│  └── Well under 30-minute constraint                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Constraints Compliance

### Constraint 1: Model Size (Max 1 GB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CHECKPOINT SIZE ANALYSIS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Option A: Save only LoRA + Head (Recommended)                             │
│  ├── LoRA adapters: ~5 MB                                                  │
│  ├── Prediction head: ~18 MB (FP32) or ~9 MB (FP16)                       │
│  ├── Vocabulary: ~0.5 MB                                                   │
│  ├── TOTAL: ~15-25 MB                                                      │
│  └── ✅ Well under 1 GB limit                                              │
│                                                                             │
│  Note: Base CANINE model is loaded from HuggingFace Hub at inference      │
│  (downloaded during Docker build, not counted as checkpoint)              │
│                                                                             │
│  Option B: Save full merged model                                          │
│  ├── Full CANINE + head: ~275 MB (FP16)                                   │
│  └── ✅ Still under 1 GB limit                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Constraint 2: Source Code Size (Max 1 MB)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SOURCE CODE SIZE ESTIMATE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  src/                                                                       │
│  ├── myprogram.py          ~10 KB   (main script)                          │
│  ├── model.py              ~5 KB    (model definition)                     │
│  ├── data.py               ~5 KB    (data loading)                         │
│  ├── train.py              ~8 KB    (training logic)                       │
│  ├── predict.py            ~3 KB    (inference)                            │
│  ├── utils.py              ~3 KB    (utilities)                            │
│  └── predict.sh            ~0.5 KB  (shell script)                         │
│  ────────────────────────────────────────────────                          │
│  TOTAL: ~35 KB                                                              │
│  ✅ Well under 1 MB limit                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Constraint 3: Runtime (Max 30 minutes)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME ANALYSIS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model Loading:                                                             │
│  ├── Load CANINE from cache: ~5 seconds                                   │
│  ├── Load LoRA adapters: ~1 second                                        │
│  ├── Load head weights: ~1 second                                         │
│  └── Subtotal: ~7 seconds                                                  │
│                                                                             │
│  Inference (assuming ~2000 test examples):                                 │
│  ├── Tokenization: ~2 seconds                                             │
│  ├── Forward passes (batched): ~10 seconds                                │
│  ├── Top-3 selection: ~1 second                                           │
│  └── Subtotal: ~13 seconds                                                 │
│                                                                             │
│  File I/O:                                                                  │
│  ├── Read input file: ~1 second                                           │
│  ├── Write output file: ~1 second                                         │
│  └── Subtotal: ~2 seconds                                                  │
│                                                                             │
│  TOTAL ESTIMATED: ~22 seconds                                              │
│  ✅ Well under 30-minute limit                                              │
│                                                                             │
│  Safety margin: Even 10x slowdown → 220 seconds ≈ 4 minutes               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Constraint 4: Docker Compatibility

```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# Install dependencies
RUN pip install transformers==4.36.0 peft==0.7.0 torch

# Pre-download CANINE model (cached in Docker image)
RUN python -c "from transformers import CanineModel; CanineModel.from_pretrained('google/canine-c')"
```

---

## 12. Evaluation Strategy

### Metrics

```
Primary Metric: Top-3 Accuracy (case-insensitive)

Formula:
  accuracy = (# correct predictions) / (# total predictions)

Where a prediction is correct if:
  gold_char.lower() in predicted_chars.lower()

Example:
  Gold: "a"
  Prediction: "Aes"
  Result: ✅ Correct (because 'a' in "aes".lower())
```

### Per-Language Analysis

```python
# Evaluation breakdown by language
def evaluate_by_language(predictions, golds, languages):
    results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, gold, lang in zip(predictions, golds, languages):
        results[lang]['total'] += 1
        if gold.lower() in pred.lower():
            results[lang]['correct'] += 1
    
    for lang, counts in results.items():
        acc = counts['correct'] / counts['total']
        print(f"{lang}: {counts['correct']}/{counts['total']} = {acc:.2%}")
```

### Expected Performance by Language

| Language | Script | Expected Accuracy | Notes |
|----------|--------|-------------------|-------|
| English | Latin | 75-85% | Strong pretraining coverage |
| French | Latin | 75-85% | Similar to English |
| German | Latin | 70-80% | Compound words may be harder |
| Italian | Latin | 75-85% | Similar to French |
| Russian | Cyrillic | 70-80% | Good pretraining coverage |
| Chinese | CJK | 60-75% | Large character set |
| Japanese | Mixed | 60-75% | Multiple scripts |
| Korean | Hangul | 65-75% | Syllable-based |
| Hindi | Devanagari | 60-75% | Less pretraining data |
| Arabic | Arabic | 60-75% | RTL, complex morphology |

### Ablation Studies

| Experiment | Purpose |
|------------|---------|
| Head-only vs LoRA+Head | Measure benefit of encoder adaptation |
| LoRA rank (r=8,16,32) | Find optimal adaptation capacity |
| Padding-only vs alternate masking | Confirm that prefix-only training does not require causal masking |
| Training data size | Measure data efficiency |
| Per-language fine-tuning | Check if specialization helps |

---

## 13. Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION TIMELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WEEK 1 (Feb 3-9): Foundation                                              │
│  ├── Day 1-2: Environment setup                                           │
│  │   ├── Set up cloud GPU instance                                        │
│  │   ├── Install dependencies (transformers, peft, torch)                │
│  │   └── Verify CANINE model loads correctly                             │
│  ├── Day 3-4: Baseline implementation                                     │
│  │   ├── Implement N-gram baseline (insurance)                           │
│  │   ├── Test on open-dev set                                            │
│  │   └── Establish baseline accuracy                                      │
│  ├── Day 5-6: CANINE + Head implementation                                │
│  │   ├── Implement CanineLoRACharPredictor class                          │
│  │   ├── Add prefix-only next-character prediction head                   │
│  │   └── Verify forward pass works                                        │
│  └── Day 7: First training run                                            │
│      ├── Small-scale test (subset of data)                                │
│      └── Debug any issues                                                  │
│                                                                             │
│  WEEK 2 (Feb 10-16): Data & Training                                       │
│  ├── Day 1-2: Data collection                                             │
│  │   ├── Download NASA transcripts                                        │
│  │   └── Set up translation pipeline                                      │
│  ├── Day 3: Data processing                                               │
│  │   ├── Clean and preprocess text                                        │
│  │   ├── Generate translations                                            │
│  │   └── Build character vocabulary                                       │
│  ├── Day 4-5: Full training                                               │
│  │   ├── Add LoRA adapters                                                │
│  │   ├── Train on full dataset                                            │
│  │   └── Monitor training metrics                                         │
│  ├── Day 6: Evaluation                                                     │
│  │   ├── Evaluate on open-dev                                             │
│  │   └── Analyze per-language performance                                 │
│  └── Day 7: Hyperparameter tuning                                         │
│      ├── Experiment with LoRA rank                                        │
│      └── Adjust learning rates                                            │
│                                                                             │
│  WEEK 3 (Feb 17-23): Polish & Submit                                       │
│  ├── Day 1-2: Optimization                                                │
│  │   ├── Implement batch inference                                        │
│  │   ├── Add FP16 inference                                               │
│  │   └── Profile and optimize                                             │
│  ├── Day 3-4: Containerization                                            │
│  │   ├── Write Dockerfile                                                 │
│  │   ├── Test Docker build                                                │
│  │   └── Verify end-to-end pipeline                                       │
│  ├── Day 5: Integration testing                                           │
│  │   ├── Test with grader/grade.sh                                       │
│  │   └── Fix any issues                                                   │
│  ├── Day 6: Buffer                                                        │
│  │   └── Address unexpected issues                                        │
│  └── Day 7: Submit                                                        │
│      ├── Package submission (submit.sh)                                   │
│      ├── Final verification                                               │
│      └── Submit to LEARN                                                  │
│                                                                             │
│  POST-MIDTERM (Feb 24+): Iteration                                         │
│  ├── Analyze validation feedback                                          │
│  ├── Collect additional training data                                     │
│  ├── Experiment with model improvements                                   │
│  │   ├── Larger LoRA rank                                                 │
│  │   ├── Ensemble with N-gram                                             │
│  │   └── Language-specific fine-tuning                                   │
│  └── Prepare final report                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 14. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CANINE underperforms | Medium | High | N-gram baseline as fallback |
| Alternative masking reduces quality | Low | Medium | Keep padding-only prefix modeling unless a clear benefit is shown |
| Training data insufficient | Medium | Medium | Augment with Wikipedia, synthetic data |
| Docker build fails | Low | High | Test containerization early (Week 2) |
| Inference too slow | Low | Medium | Batch processing, FP16, LoRA merge |
| GPU memory issues | Low | Medium | Gradient checkpointing, smaller batch |
| Translation quality poor | Medium | Low | Use multiple translation services |
| Overfitting | Medium | Medium | LoRA regularization, dropout, early stopping |

### Fallback Plan

If CANINE + LoRA fails to meet performance targets:

```
Primary: CANINE + LoRA + Head (expected: 70-80%)
    │
    ▼ (if <60% accuracy)
Fallback 1: CANINE frozen + Head only (expected: 60-70%)
    │
    ▼ (if <50% accuracy)
Fallback 2: N-gram with Kneser-Ney smoothing (expected: 45-55%)
    │
    ▼ (if all else fails)
Fallback 3: Frequency-based baseline (expected: 30-40%)
```

---

## 15. Alternatives Considered

### Why Not These Approaches?

| Alternative | Reason for Rejection |
|-------------|---------------------|
| **ByT5 (byte-level)** | UTF-8 fragmentation for CJK; multi-byte predictions needed |
| **GPT-2 (subword)** | Cannot directly predict characters; tokenization misalignment |
| **H-MoE-Mamba** | No pretraining; requires massive compute; high implementation risk |
| **Char-GPT from scratch** | No pretrained character-level models; data hungry |
| **XLM-RoBERTa + head** | Subword input; not character-native |
| **LSTM/GRU** | Limited context; no pretrained multilingual knowledge |
| **Pure N-gram** | Limited context (5-8 chars); no semantic understanding |

### Why CANINE + LoRA?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY CANINE + LoRA IS OPTIMAL                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ Character-level: Native Unicode character processing                   │
│  ✅ Pretrained: 104 languages (covers all 10 target languages)             │
│  ✅ Efficient: LoRA trains only 4% of parameters                           │
│  ✅ Small checkpoint: ~15 MB (vs 1 GB limit)                               │
│  ✅ Fast inference: Single forward pass per prediction                     │
│  ✅ Low risk: Proven fine-tuning methodology                               │
│  ✅ Adaptable: Prefix-only supervision avoids future-target leakage        │
│                                                                             │
│  This is the ONLY approach that satisfies ALL constraints while           │
│  leveraging pretrained multilingual character-level knowledge.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: File Structure

```
project/
├── src/
│   ├── myprogram.py        # Main entry point
│   ├── model.py            # CanineLoRACharPredictor class
│   ├── data.py             # Dataset and DataLoader
│   ├── train.py            # Training logic
│   ├── predict.py          # Inference logic
│   ├── utils.py            # Utility functions
│   └── predict.sh          # Shell script for inference
├── work/
│   ├── canine_lora_adapters/   # LoRA weights
│   ├── char_head.pt            # Prediction head weights
│   ├── vocab.json              # Output character vocabulary
│   └── model_config.json       # Serialized ModelConfig
├── data/
│   ├── open-dev/               # Provided development data
│   └── training/               # Collected training data
├── grader/
│   ├── grade.py                # Evaluation script
│   └── grade.sh                # Grading shell script
├── Dockerfile                  # Container definition
├── PROJECT.md                  # This file
├── README.md                   # Course instructions
└── submit.sh                   # Packaging script
```

---

*Document Version: 1.0*
*Last Updated: February 2, 2026*
*Author: CS498 NLP Course Project*
