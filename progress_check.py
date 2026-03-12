#!/usr/bin/env python3
"""
MESSI — Progress Checker
Run this from the project root to assess how much of the pipeline
is working end-to-end on your machine.

    python progress_check.py

It will test each layer independently and print a coloured report.
No training data or checkpoint is required.
"""

import sys
import traceback
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

STATUS = {"pass": 0, "fail": 0, "warn": 0}

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def ok(msg):
    STATUS["pass"] += 1
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg, err=""):
    STATUS["fail"] += 1
    short = str(err).split("\n")[0][:120] if err else ""
    print(f"  {RED}✗{RESET} {msg}")
    if short:
        print(f"    {RED}→ {short}{RESET}")


def warn(msg):
    STATUS["warn"] += 1
    print(f"  {YELLOW}⚠{RESET} {msg}")


def section(title):
    print(f"\n{BOLD}{BLUE}{'─'*55}{RESET}")
    print(f"{BOLD}{BLUE} {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*55}{RESET}")


# ────────────────────────────────────────────────────────────────
#  Phase 0 — Python & Dependencies
# ────────────────────────────────────────────────────────────────
section("Phase 0: Python Environment")

py = sys.version_info
if py >= (3, 9):
    ok(f"Python {py.major}.{py.minor}.{py.micro}")
else:
    fail(f"Python {py.major}.{py.minor} — need ≥ 3.9")

required_packages = [
    ("torch",        "PyTorch"),
    ("spacy",        "spaCy"),
    ("emoji",        "emoji"),
    ("ortools",      "OR-Tools"),
    ("numpy",        "NumPy"),
    ("sklearn",      "scikit-learn"),
    ("requests",     "requests"),
    ("jsonlines",    "jsonlines"),
    ("transformers", "Hugging Face Transformers (baseline — optional)"),
]

for pkg, label in required_packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        ok(f"{label} {ver}")
    except ImportError as e:
        if "optional" in label:
            warn(f"{label} not installed (only needed for BERT baseline)")
        else:
            fail(f"{label} — NOT INSTALLED", e)


# ────────────────────────────────────────────────────────────────
#  Layer 1 — Preprocessing
# ────────────────────────────────────────────────────────────────
section("Layer 1: Emoji-Aware Preprocessing")

try:
    from layers.layer1_preprocessing.emoji_vocab import build_vocab_from_texts, save_vocab
    texts = ["order 782 not delivered 😠", "flight UA123 delayed 😤", "payment failed 💀🔥"]
    vocab = build_vocab_from_texts(texts)
    assert "😠" in vocab and "<PAD>" in vocab and "<UNK>" in vocab
    ok(f"Emoji vocab built — {len(vocab)} entries  (😠:{vocab.get('😠')}, 🔥:{vocab.get('🔥')})")

    # Save for downstream use
    from config import EMOJI_VOCAB_PATH
    save_vocab(vocab, EMOJI_VOCAB_PATH)
    ok(f"Emoji vocab saved → {EMOJI_VOCAB_PATH}")
except Exception as e:
    fail("Emoji vocab builder", e)
    traceback.print_exc()
    vocab = {"<PAD>": 0, "<UNK>": 1}

try:
    from layers.layer1_preprocessing.tokenizer import split_emoji_tokens, build_emoji_aware_nlp, tokenize

    tokens = split_emoji_tokens("delayed😤arrival")
    assert "😤" in tokens, f"emoji not split, got {tokens}"
    ok(f"Emoji splitter: 'delayed😤arrival' → {tokens}")

    nlp = build_emoji_aware_nlp()
    result = tokenize("order #782 not delivered 😠", nlp)
    ok(f"spaCy tokenizer: {result}")
except Exception as e:
    fail("Emoji tokenizer / spaCy pipeline", e)
    nlp = None

try:
    from layers.layer1_preprocessing.embeddings import CombinedEmbedding, EmbeddingExtractor
    import torch

    emb_module = CombinedEmbedding(emoji_vocab=vocab)
    extractor  = EmbeddingExtractor(nlp, vocab) if nlp else None

    if extractor:
        spacy_t, emoji_t = extractor.batch_extract([["order", "782", "😠"]])
        assert spacy_t.shape == (1, 3, 300), f"bad spacy_t shape {spacy_t.shape}"
        combined = emb_module(spacy_t, emoji_t)
        assert combined.shape == (1, 3, 350), f"bad combined shape {combined.shape}"
        ok(f"CombinedEmbedding: (1, 3, 350) ✓  [spaCy 300 + emoji 50 = 350]")
    else:
        warn("Skipping embedding test (spaCy not loaded)")
except Exception as e:
    fail("CombinedEmbedding", e)


# ────────────────────────────────────────────────────────────────
#  Layer 2 — BiLSTM-CRF
# ────────────────────────────────────────────────────────────────
section("Layer 2: BiLSTM-CRF Neural Model")

try:
    import torch
    from layers.layer2_bilstm_crf.crf import CRFLayer
    from config import NUM_TAGS

    crf = CRFLayer(num_tags=NUM_TAGS)
    B, L = 2, 8
    emissions = torch.randn(B, L, NUM_TAGS)
    tags  = torch.zeros(B, L, dtype=torch.long)
    mask  = torch.ones(B, L, dtype=torch.bool)

    loss = crf.neg_log_likelihood(emissions, tags, mask)
    assert not torch.isnan(loss), "NaN loss!"
    seqs = crf.viterbi(emissions, mask)
    assert len(seqs) == B and len(seqs[0]) == L
    ok(f"CRF forward + Viterbi: loss={loss.item():.4f}, decoded {B} sequences of len {L}")
except Exception as e:
    fail("CRF Layer", e)
    traceback.print_exc()

try:
    from layers.layer2_bilstm_crf.model import BiLSTMCRF
    from config import EMBEDDING_DIM

    model = BiLSTMCRF(emoji_vocab=vocab)
    params = sum(p.numel() for p in model.parameters())
    ok(f"BiLSTMCRF built — {params:,} parameters")

    # Forward pass
    B, L = 1, 6
    sv   = torch.randn(B, L, 300)
    ei   = torch.zeros(B, L, dtype=torch.long)
    ta   = torch.zeros(B, L, dtype=torch.long)
    msk  = torch.ones(B, L, dtype=torch.bool)

    loss   = model(sv, ei, ta, msk)
    decode = model.decode(sv, ei, msk)
    assert not torch.isnan(loss)
    assert len(decode[0]) == L
    ok(f"BiLSTMCRF forward: loss={loss.item():.4f} | decode: {[str(t) for t in decode[0]]}")
except Exception as e:
    fail("BiLSTMCRF Model", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Layer 3 — ILP Validator
# ────────────────────────────────────────────────────────────────
section("Layer 3: ILP Symbolic Validator (OR-Tools)")

try:
    from layers.layer3_ilp.constraints import ConstraintValidator, extract_spans_from_bio

    cv = ConstraintValidator()
    assert cv.is_valid("FLIGHT_ID", "UA123")  is True
    assert cv.is_valid("FLIGHT_ID", "123")    is False
    assert cv.is_valid("ORDER_ID",  "782")    is True
    assert cv.is_valid("ORDER_ID",  "abc")    is False
    assert cv.is_valid("ISSUE_TYPE","not_delivered") is True
    ok("Constraint rules: FLIGHT_ID regex, ORDER_ID numeric, ISSUE_TYPE allowlist ✓")

    spans = extract_spans_from_bio(
        ["order", "782", "not", "delivered"],
        ["O",     "B-ORDER_ID", "O", "B-ISSUE_TYPE"]
    )
    assert len(spans) == 2
    ok(f"BIO→spans: {[(s['field'], s['text']) for s in spans]}")
except Exception as e:
    fail("Constraint validator / BIO-to-span", e)
    traceback.print_exc()

try:
    from layers.layer3_ilp.solver import ILPValidator, validate_prediction

    ilp = ILPValidator()
    result = validate_prediction(
        ["order", "782", "not", "delivered", "😠"],
        ["O",     "B-ORDER_ID", "O", "B-ISSUE_TYPE", "O"],
        validator=ilp,
    )
    assert result["validation_status"] == "Passed ILP Constraints"
    ok(f"ILP solve: {result['validation_status']} | record={result['record']}")

    # Should FAIL for invalid flight ID "123"
    bad = validate_prediction(["flight", "123"], ["O", "B-FLIGHT_ID"], validator=ilp)
    assert bad["record"].get("FLIGHT_ID") is None
    ok("ILP correctly rejects invalid FLIGHT_ID '123' (not [A-Z]{2}\\d{3,4})")
except Exception as e:
    fail("ILP Solver", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Layer 4 — Uncertainty Estimation
# ────────────────────────────────────────────────────────────────
section("Layer 4: MC Dropout Uncertainty Estimation")

try:
    import math
    from layers.layer4_uncertainty.entropy import predictive_entropy
    from layers.layer4_uncertainty.mc_dropout import (
        compute_confidence, overall_entropy, should_route_to_human
    )
    from config import ENTROPY_THRESHOLD

    # Perfect certainty → H=0
    H_certain = predictive_entropy({"B-ORDER_ID": 10}, T=10)
    assert abs(H_certain) < 1e-6
    ok(f"Zero entropy when perfectly certain: H={H_certain:.6f} ✓")

    # MC Dropout on the model built in Layer 2
    from layers.layer4_uncertainty.mc_dropout import mc_dropout_predict
    model_l4 = BiLSTMCRF(emoji_vocab=vocab)
    sv   = torch.randn(1, 5, 300)
    ei   = torch.zeros(1, 5, dtype=torch.long)
    msk  = torch.ones(1, 5, dtype=torch.bool)
    best_tags, entropies = mc_dropout_predict(model_l4, sv, ei, msk, T=5)
    confs = compute_confidence(entropies)
    oe    = overall_entropy(entropies)
    ok(f"MC Dropout (T=5): entropies={entropies} | confidences={confs} | overall_entropy={oe}")

    route = should_route_to_human({"ORDER_ID": ENTROPY_THRESHOLD + 0.1})
    assert route is True
    ok(f"Human routing triggered at H > {ENTROPY_THRESHOLD} ✓")
except Exception as e:
    fail("MC Dropout / Uncertainty", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Layer 5 — Decision Engine
# ────────────────────────────────────────────────────────────────
section("Layer 5: Decision & Automation Engine")

try:
    from layers.layer5_decision.engine import DecisionEngine

    engine = DecisionEngine()
    decision = engine.decide(
        record={"ORDER_ID": "782", "ISSUE_TYPE": "not_delivered"},
        confidences={"ORDER_ID": 0.98, "ISSUE_TYPE": 0.91},
        entropies={"ORDER_ID": 0.02, "ISSUE_TYPE": 0.05},
        raw_text="order #782 not delivered 😠",
        validation_status="Passed ILP Constraints",
    )
    assert decision["routing"] == "automated"
    assert decision["urgency"] == "high"   # anger emoji triggers high
    assert "zendesk" in decision["action_triggered"]
    ok(f"Decision: urgency={decision['urgency']} | action={decision['action_triggered']} | routing={decision['routing']}")

    # High entropy → human review
    uncertain = engine.decide(
        record={"ORDER_ID": "999"},
        confidences={"ORDER_ID": 0.60},
        entropies={"ORDER_ID": 0.9},
        raw_text="unclear text",
        validation_status="Passed ILP Constraints",
    )
    assert uncertain["routing"] == "human_review"
    ok(f"High entropy → routing=human_review ✓")
except Exception as e:
    fail("Decision Engine", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Layer 6 — Payload Builder
# ────────────────────────────────────────────────────────────────
section("Layer 6: API Payload Builder")

try:
    from layers.layer6_api.payload_builder import build_output_payload, dispatch_action

    out = build_output_payload(decision, raw_text="order #782 not delivered 😠")
    # Verify Blueprint §5 schema
    for key in ["validation_status", "urgency", "action_triggered", "routing", "confidence"]:
        assert key in out, f"Missing key: {key}"
    assert "overall_entropy" in out["confidence"]
    assert "entity_id" in out   # ORDER_ID remapped
    ok(f"Output schema valid ✓")
    ok(f"entity_id={out['entity_id']} | issue_type={out.get('issue_type')} | urgency={out['urgency']}")

    # Dry-run dispatch
    out2 = dispatch_action(out.copy(), dry_run=True)
    assert out2["api_response"]["status"] == "dry_run"
    ok(f"Dry-run dispatch: {out2['api_response']['message']}")
except Exception as e:
    fail("Payload builder / dispatch", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Full End-to-End (no checkpoint)
# ────────────────────────────────────────────────────────────────
section("End-to-End Pipeline (untrained model)")

try:
    from main import load_pipeline, predict

    pipeline = load_pipeline()
    result = predict("order #782 not delivered 😠", pipeline, dry_run=True)

    import json
    print(f"\n  {BOLD}Input:{RESET}  order #782 not delivered 😠")
    print(f"  {BOLD}Output:{RESET}")
    print("  " + json.dumps(result, indent=4, ensure_ascii=False).replace("\n", "\n  "))
    ok("Full pipeline ran successfully (untrained — results will improve after training)")
except Exception as e:
    fail("End-to-end pipeline", e)
    traceback.print_exc()


# ────────────────────────────────────────────────────────────────
#  Summary
# ────────────────────────────────────────────────────────────────
section("Progress Summary")
total = STATUS["pass"] + STATUS["fail"] + STATUS["warn"]
pct   = int(STATUS["pass"] / max(total, 1) * 100)
bar   = ("█" * (pct // 5)).ljust(20)
print(f"\n  {GREEN}{bar}{RESET} {pct}%")
print(f"  {GREEN}✓ {STATUS['pass']} passed{RESET}  "
      f"{RED}✗ {STATUS['fail']} failed{RESET}  "
      f"{YELLOW}⚠ {STATUS['warn']} warnings{RESET}")

if STATUS["fail"] == 0:
    print(f"\n  {BOLD}{GREEN}🎉 All systems operational! Run:{RESET}")
    print(f"     python main.py --text \"order #782 not delivered 😠\" --pretty")
else:
    print(f"\n  {BOLD}{YELLOW}Fix the failures above, then re-run: python progress_check.py{RESET}")

print()
