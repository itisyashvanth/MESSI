# MESSI — Mathematically Enhanced Structuring of Semi-Structured Inputs

> **A Neuro-Symbolic Information Extraction & Decision Automation Platform**

---

## What MESSI Does

Takes messy, emoji-laden customer messages and turns them into structured, validated, actionable records — fully offline.

```
"😠 order #4540 not delivered again asap"
         ↓
  RECORD:    { ORDER_ID: "4540", ISSUE_TYPE: "not delivered" }
  URGENCY:   HIGH 🔴
  ACTION:    zendesk_ticket
  ROUTING:   automated  (confidence 0.94)
```

---

## 6-Layer Pipeline

| Layer | File | What it does |
|-------|------|-------------|
| 1 | `preprocessing.py` | Emoji-aware tokeniser + spaCy(300d) + emoji(50d) = ℝ³⁵⁰ embedding |
| 2 | `model.py` | 3-layer Bidirectional LSTM + CRF decoder + Char-CNN encoder |
| 3 | `ilp.py` | OR-Tools CP-SAT ILP validator (regex, allowlist, cardinality, no-overlap) |
| 4 | `uncertainty.py` | Monte Carlo Dropout (T=10 passes) → predictive entropy per field |
| 5 | `uncertainty.py` | Rule-based Decision Engine → 3-tier urgency + action routing |
| 6 | `api.py` | Lazy-import dispatcher → Zendesk / Salesforce / Slack / Twilio / Firebase / PostgreSQL |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 2. One-command full pipeline

```bash
python run_all.py
```

This runs all 5 stages in order:
```
✓ Data Generation   → 5,000 BIO-tagged samples (e-commerce + aviation)
✓ Model Training    → BiLSTM-CRF with char-CNN, warmup schedule, 60 epochs
✓ Evaluation        → Token-level + span-level F1, Cohen's κ
✓ Benchmark         → Parameters, latency, throughput, accuracy
✓ Live Demo         → 12 showcase messages through full pipeline
```

**Quick mode (5 epochs, 800 samples — for testing):**

```bash
python run_all.py --quick
```

**Skip retraining (use existing checkpoint):**

```bash
python run_all.py --skip-train
```

### 3. Live demo

```bash
# Run 12 showcase messages (both domains)
python demo.py

# Your own message
python demo.py --text "😠 order #782 payment failed send help"
```

### 4. Run individual stages

```bash
# Generate data
python data_utils.py --n-ec 3000 --n-av 2000

# Train
python train.py --epochs 60

# Evaluate (token + span F1)
python evaluate.py --test data/annotated/test.jsonl

# Benchmark (efficiency + accuracy)
python benchmark.py --epochs 20 --samples 1000

# Single inference
python main.py --text "flight UA4821 delayed 🔥"
```

---

## Emoji Vocabulary — 7 Categories, 45 Unique Emojis

| Category | Emojis | Urgency |
|----------|--------|---------|
| **ANGER** (most frequent) | 😠 😡 🤬 💀 🔥 | 🔴 HIGH |
| **ALERT** (most frequent) | ⚠️ 🚨 🔔 ⏰ 📢 🆘 | 🔴 HIGH |
| **MILD** (most frequent) | 😑 🙄 😒 😤 | 🟡 MEDIUM |
| SADNESS | 😢 😭 💔 😔 😞 😿 🥺 😩 😫 | 🟡 MEDIUM |
| SARCASM | 😏 🤡 🥴 💅 | 🟡 MEDIUM |
| POSITIVE | 👍 😊 🙏 ✅ 🎉 😄 🤗 💯 🙌 | 🟢 LOW |
| NEUTRAL | 🤷 😐 🤔 🫤 😶 🧐 🫥 | 🟢 LOW |

---

## Entity Schema (BIO Tags)

### E-Commerce Domain
| Tag | Example |
|-----|---------|
| `B-ORDER_ID` / `I-ORDER_ID` | `#4540`, `7821` |
| `B-ISSUE_TYPE` / `I-ISSUE_TYPE` | `payment failed`, `not delivered` |

### Aviation Domain
| Tag | Example |
|-----|---------|
| `B-FLIGHT_ID` / `I-FLIGHT_ID` | `UA4821`, `DL9902` |
| `B-EVENT` / `I-EVENT` | `delayed`, `has been cancelled` |
| `B-TIME` / `I-TIME` | `3 hours`, `by 5pm` |

---

## Model Architecture (v2)

```
Input Tokens
    ├── spaCy en_core_web_md (frozen, 300d) ─┐
    ├── Trainable emoji embedding (50d)      ├─ concat → ℝ^350 → Dropout(0.25)
    └── Char-CNN (Conv1d, 64 filters, k=3)  ─┘         → ℝ^414

    → 3-layer BiLSTM (hidden=320, bidirectional) → ℝ^640
    → Linear projection → ℝ^K (K = 11 tags)
    → CRF Viterbi decoder (with label smoothing α=0.08)

Parameters:   ~4.2M trainable
Latency:      ~3–5 ms / sample (CPU)
Throughput:   ~250 samples/sec (batch=32, CPU)
```

---

## Training Hyperparameters (v2)

| Setting | Value |
|---------|-------|
| Optimiser | Adam (weight_decay=1e-5) |
| Learning rate | 5e-4 |
| Schedule | Linear warmup (2 epochs) → Cosine decay |
| Epochs | 60 (early stopping patience=10) |
| Batch size | 32 |
| Gradient clipping | 5.0 |
| Label smoothing | 0.08 |

---

## Expected Benchmark Results

| Metric | Value |
|--------|-------|
| Macro Token F1 | 0.87–0.92 |
| Span-level F1 | 0.84–0.90 |
| Cohen's κ | 0.82–0.88 |
| Inference latency | < 5 ms/sample |
| Throughput | > 200 samples/sec |

---

## Project Files

```
messi/
├── config.py          # All hyperparameters, paths, thresholds
├── preprocessing.py   # Layer 1: tokeniser + embeddings
├── model.py           # Layer 2: BiLSTM-CRF + Char-CNN
├── ilp.py             # Layer 3: ILP symbolic validator
├── uncertainty.py     # Layers 4+5: MC Dropout + Decision Engine
├── api.py             # Layer 6: API dispatch
├── data_utils.py      # Synthetic data generator (2 domains, 45 emojis)
├── train.py           # Training loop (AMP, warmup, early stopping)
├── evaluate.py        # Token + span F1, Cohen's κ, error analysis
├── benchmark.py       # Efficiency + accuracy benchmark
├── demo.py            # 🎯 Live inference showcase
├── run_all.py         # 🚀 One-command pipeline runner
├── main.py            # End-to-end inference entry point
├── baselines.py       # spaCy NER + BERT baselines
└── tests/
    └── test_messi.py  # Consolidated test suite
```

---

## Environment Variables (API Integration)

```bash
export ZENDESK_URL="https://your-domain.zendesk.com"
export ZENDESK_EMAIL="admin@example.com"
export ZENDESK_TOKEN="..."
export SALESFORCE_URL="https://your-domain.salesforce.com"
export SALESFORCE_TOKEN="..."
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export POSTGRES_DSN="postgresql://user:pass@localhost:5432/messi"
```

APIs run in **dry-run mode by default** — set `--live` in demo.py to dispatch real actions.

---

## Academic References

- Ma & Hovy (2016) — *End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF*
- Lafferty et al. (2001) — *Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data*
- Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation*
- Müller-Eising (2019) — *ILP-Based Constraint Solving for NLP*
