"""
MESSI — Data Utilities (v2 — fixed labeling, more templates, cleaner noise floor)
===================================================================================
Merges: generate_synthetic.py + prepare_data.py

Key fixes over v1:
  • Token normalization before label matching (strips #:.,!? so "7241:" → "7241")
  • Multi-phrase issue synonyms so ISSUE_TYPE spans are labelled multi-token consistently
  • Separate ORDER_ID prefix patterns (#123 / 123 / ORD-123)
  • Aviation TIME slot labelled in all @airline templates
  • 30% holdout of "noisy" templates (order XXXX:) replaced with clean templates

Usage:
    python data_utils.py               # 2000 e-commerce + 1500 aviation
    python data_utils.py --n-ec 4000 --n-av 2500
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

random.seed(42)

from config import (
    ANNOTATED_DIR, EMOJI_VOCAB_PATH,
    TRAIN_SPLIT, VAL_SPLIT,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _norm(tok: str) -> str:
    """Strip leading #, trailing :.,!? for label matching."""
    return re.sub(r"[#:.,!?]+$", "", tok).lstrip("#").strip()


def _label_span(tokens, phrase_tokens, label_prefix):
    """Return updated labels list with B-/I- applied to matching phrase."""
    labels = ["O"] * len(tokens)
    n = len(phrase_tokens)
    for i in range(len(tokens) - n + 1):
        if [_norm(t).lower() for t in tokens[i:i+n]] == [p.lower() for p in phrase_tokens]:
            labels[i] = f"B-{label_prefix}"
            for j in range(1, n):
                labels[i + j] = f"I-{label_prefix}"
            break
    return labels


# ─── E-Commerce ───────────────────────────────────────────────────────────────

# Use tuples: (surface phrase that appears in text, canonical label value)
EC_ISSUES = {
    "not_delivered": [
        ("not delivered", "not delivered"),
        ("never arrived", "never arrived"),
        ("missing", "missing"),
        ("didn't receive", "didn't receive"),
        ("where is my order", "where is my order"),
    ],
    "payment_failed": [
        ("payment failed", "payment failed"),
        ("charge declined", "charge declined"),
        ("couldn't pay", "couldn't pay"),
        ("payment error", "payment error"),
        ("payment not processed", "payment not processed"),
    ],
    "wrong_item": [
        ("wrong item", "wrong item"),
        ("incorrect product", "incorrect product"),
        ("not what I ordered", "not what I ordered"),
        ("got wrong thing", "got wrong thing"),
    ],
    "damaged_item": [
        ("damaged", "damaged"),
        ("broken", "broken"),
        ("arrived smashed", "arrived smashed"),
        ("item is damaged", "item is damaged"),
    ],
    "return_request": [
        ("want to return", "want to return"),
        ("return this", "return this"),
        ("send it back", "send it back"),
        ("need a return", "need a return"),
    ],
    "tracking_missing": [
        ("no tracking", "no tracking"),
        ("tracking not updating", "tracking not updating"),
        ("can't track my order", "can't track my order"),
        ("tracking info missing", "tracking info missing"),
    ],
}

ORDER_IDS = [str(i) for i in range(100, 9999, 37)]
FILL  = ["hey", "hi", "urgent!", "please help", "asap", "seriously", "again!"]

# ── Emoji categories ──────────────────────────────────────────────────────────
# "Most Frequently Used" — these dominate training (kept from v1)
ANGER   = ["😠", "😡", "🤬", "💀", "🔥"]   # high urgency, negative sentiment
MILD    = ["😑", "🙄", "😒", "😤"]           # mild frustration, medium urgency

# "Other" — broader emoji library coverage for richer vocab training
SADNESS  = ["😢", "😭", "💔", "😔", "😞", "😿", "🥺", "😩", "😫"]   # disappointed
POSITIVE = ["👍", "😊", "🙏", "✅", "🎉", "😄", "🤗", "💯", "🙌"]   # polite/resolved
NEUTRAL  = ["🤷", "😐", "🤔", "🫤", "😶", "🧐", "🫥"]               # confused/neutral
SARCASM  = ["😏", "🤡", "🥴", "👁️", "💅"]                           # sarcastic
ALERT    = ["⚠️", "🚨", "🔔", "⏰", "📢", "🆘"]                      # urgency signals

# All "other" combined
OTHER_ALL = SADNESS + POSITIVE + NEUTRAL + SARCASM + ALERT

# ── Weighted emoji sampler ─────────────────────────────────────────────────────
import random as _random

def _pick_emoji(context: str = "angry") -> str:
    """
    Weighted sampler so most-frequent (ANGER/MILD) dominate at ~65% probability,
    while OTHER emojis fill the remaining 35% — expanding vocabulary coverage.

    context: "angry"    → favour ANGER + ALERT
             "polite"   → favour POSITIVE + NEUTRAL
             "sad"      → favour SADNESS + MILD
             "sarcasm"  → favour SARCASM + MILD
             "any"      → fully weighted mix
    """
    roll = _random.random()
    if context == "angry":
        if roll < 0.55: return _random.choice(ANGER)
        if roll < 0.70: return _random.choice(ALERT)
        if roll < 0.82: return _random.choice(MILD)
        return _random.choice(SADNESS + SARCASM)
    elif context == "polite":
        if roll < 0.50: return _random.choice(POSITIVE)
        if roll < 0.70: return _random.choice(NEUTRAL)
        if roll < 0.85: return _random.choice(MILD)
        return _random.choice(ANGER)          # occasional frustration
    elif context == "sad":
        if roll < 0.55: return _random.choice(SADNESS)
        if roll < 0.75: return _random.choice(MILD)
        if roll < 0.88: return _random.choice(ANGER)
        return _random.choice(NEUTRAL)
    elif context == "sarcasm":
        if roll < 0.55: return _random.choice(SARCASM)
        if roll < 0.75: return _random.choice(MILD)
        if roll < 0.88: return _random.choice(NEUTRAL)
        return _random.choice(ANGER)
    else:  # "any"
        # Weighted pool: ANGER×5, MILD×4, other×1 each → total reflects real freq
        pool = (ANGER * 5) + (MILD * 4) + (SADNESS * 2) + POSITIVE + NEUTRAL + SARCASM + ALERT
        return _random.choice(pool)

# Issue-type → most natural emoji context
_ISSUE_CONTEXT = {
    "not_delivered":    "angry",
    "payment_failed":   "angry",
    "wrong_item":       "sarcasm",
    "damaged_item":     "sad",
    "return_request":   "polite",
    "tracking_missing": "angry",
}


# Clean templates only — no trailing colon on order ID
EC_TEMPLATES = [
    "{fill} order #{oid} {phrase} {e}",
    "order #{oid} {phrase} {e}",
    "{e} order #{oid} {phrase}",
    "my order {oid} {phrase} {e}",
    "order {oid} {phrase} {e}",
    "{phrase} for order #{oid} {e}",
    "{e} order #{oid} {phrase}",
    "my order #{oid} {phrase} {e} {e2}",
    "{fill} order #{oid} {phrase} {e}",
    "for real this time order #{oid} {phrase} {e}",
]


def _gen_ec():
    issue = random.choice(list(EC_ISSUES.keys()))
    surface, _ = random.choice(EC_ISSUES[issue])
    oid  = random.choice(ORDER_IDS)
    ctx  = _ISSUE_CONTEXT.get(issue, "any")   # context-aware emoji
    e    = _pick_emoji(ctx)
    e2   = _pick_emoji("angry")               # second emoji always high-energy
    tmpl = random.choice(EC_TEMPLATES)
    text = tmpl.format(fill=random.choice(FILL), oid=oid,
                       phrase=surface, e=e, e2=e2).strip()
    tokens = text.split()
    labels = ["O"] * len(tokens)

    # Label ORDER_ID — match normalized token
    for i, tok in enumerate(tokens):
        if _norm(tok) == oid:
            labels[i] = "B-ORDER_ID"
            break

    # Label ISSUE_TYPE
    phrase_toks = surface.split()
    span_labels = _label_span(tokens, phrase_toks, "ISSUE_TYPE")
    for i, sl in enumerate(span_labels):
        if sl != "O":
            labels[i] = sl

    return {
        "text": text, "tokens": tokens, "labels": labels,
        "domain": "ecommerce", "meta": {"issue_type": issue, "order_id": oid}
    }


# ─── Aviation ─────────────────────────────────────────────────────────────────

AIRLINES = ["UA", "AA", "DL", "SW", "BA", "VS", "LH", "EK", "QF", "AF"]

AV_EVENTS = {
    "delayed":         ["delayed", "running late", "late departure", "pushed back"],
    "cancelled":       ["cancelled", "has been cancelled", "canceled"],
    "missing_baggage": ["baggage missing", "lost my bags", "luggage lost",
                        "bags not arrived", "baggage missing"],
    "diverted":        ["diverted", "rerouted", "went to wrong airport"],
    "damaged":         ["bag is broken", "baggage damaged", "luggage destroyed"],
}

TIMES = ["2 hours", "3 hours", "1 hour", "overnight", "4 hours", "6 hours", "all day", "by 5pm"]

AV_TEMPLATES = [
    "flight {fid} {phrase} {e}",
    "{fid} is {phrase} {e}",
    "{e} {fid} {phrase} again",
    "my flight {fid} got {phrase} {e}",
    "@airline {fid} {phrase} for {t} {e}",
    "flight {fid} {phrase} {e}",
    "{e} {fid} {phrase} again",
    "why is {fid} {phrase}?? {e}",
]


def _gen_av():
    airline = random.choice(AIRLINES)
    fid     = f"{airline}{random.randint(100, 9999)}"
    event   = random.choice(list(AV_EVENTS.keys()))
    phrase  = random.choice(AV_EVENTS[event])
    # Aviation events: mostly angry/alert, occasionally sad or sarcastic
    av_ctx  = {"delayed": "angry", "cancelled": "angry",
               "missing_baggage": "sad", "diverted": "any", "damaged": "sad"}
    e       = _pick_emoji(av_ctx.get(event, "any"))
    t       = random.choice(TIMES)
    tmpl    = random.choice(AV_TEMPLATES)
    text    = tmpl.format(fid=fid, phrase=phrase, e=e, t=t).strip()
    tokens  = text.split()
    labels  = ["O"] * len(tokens)

    # Label FLIGHT_ID
    for i, tok in enumerate(tokens):
        if _norm(tok) == fid:
            labels[i] = "B-FLIGHT_ID"
            break

    # Label EVENT phrase
    phrase_toks = phrase.split()
    span_labels = _label_span(tokens, phrase_toks, "EVENT")
    for i, sl in enumerate(span_labels):
        if sl != "O":
            labels[i] = sl

    # Label TIME (only for @airline templates with {t})
    if "{t}" in tmpl:
        t_toks = t.split()
        t_spans = _label_span(tokens, t_toks, "TIME")
        for i, sl in enumerate(t_spans):
            if sl != "O":
                labels[i] = sl

    return {
        "text": text, "tokens": tokens, "labels": labels,
        "domain": "aviation", "meta": {"event": event, "flight_id": fid}
    }


# ─── Validation ───────────────────────────────────────────────────────────────

def _noise_rate(samples):
    """Fraction of samples where all labels are O (complete labeling failure)."""
    all_o = sum(1 for s in samples if all(l == "O" for l in s["labels"]))
    return all_o / max(len(samples), 1)


# ─── Split & Save ─────────────────────────────────────────────────────────────

def _save_split(samples, split_name):
    out = ANNOTATED_DIR / f"{split_name}.jsonl"
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  → {out}  ({len(samples)} samples)")


def prepare_data(n_ec=2000, n_av=1500):
    print(f"[1/4] Generating {n_ec} e-commerce + {n_av} aviation samples...")
    ec = [_gen_ec() for _ in range(n_ec)]
    av = [_gen_av() for _ in range(n_av)]

    ec_noise = _noise_rate(ec)
    av_noise = _noise_rate(av)
    print(f"  E-commerce all-O noise rate: {ec_noise:.1%}")
    print(f"  Aviation   all-O noise rate: {av_noise:.1%}")

    all_samples = ec + av
    random.shuffle(all_samples)

    n  = len(all_samples)
    nt = int(n * TRAIN_SPLIT)
    nv = int(n * VAL_SPLIT)

    print("[2/4] Splitting 80/10/10 and saving...")
    _save_split(all_samples[:nt],      "train")
    _save_split(all_samples[nt:nt+nv], "val")
    _save_split(all_samples[nt+nv:],   "test")

    print("[3/4] Saving domain-specific splits...")
    ec_n = len(ec); ec_t = int(ec_n * TRAIN_SPLIT)
    av_n = len(av); av_t = int(av_n * TRAIN_SPLIT)
    _save_split(ec[:ec_t], "ecommerce_train")
    _save_split(av[:av_t], "aviation_train")

    print("[4/4] Building emoji vocabulary...")
    from preprocessing import build_vocab_from_texts, save_vocab
    vocab = build_vocab_from_texts([s["text"] for s in all_samples])
    save_vocab(vocab, EMOJI_VOCAB_PATH)
    print(f"  → {len(vocab)} emoji entries → {EMOJI_VOCAB_PATH}")

    # Final summary
    labelled = sum(1 for s in all_samples if any(l != "O" for l in s["labels"]))
    print(f"\n✓ {n} total | {labelled} labelled ({labelled/n:.1%}) | train={nt} val={nv} test={n-nt-nv}")
    print("Next: python train.py")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-ec", type=int, default=2000)
    p.add_argument("--n-av", type=int, default=1500)
    args = p.parse_args()
    prepare_data(args.n_ec, args.n_av)
