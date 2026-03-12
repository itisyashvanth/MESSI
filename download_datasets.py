#!/usr/bin/env python3
"""
MESSI — Real Dataset Downloader
================================
Downloads NER datasets from HuggingFace and converts to MESSI BIO format.
No synthetic data. Uses datasets==2.18 which supports loading scripts.

Datasets:
  1. conll2003        — Classic NER (aviation-filtered: ORG→FLIGHT_ID)
  2. wnut_17          — Novel entity NER  (product→ORDER_ID, corporation→FLIGHT_ID)
  3. PolyAI/banking77 — Customer service (real text with payment/order keywords)
  4. conllpp           — CoNLL++ (higher quality relabeled CoNLL)

Usage:
    python download_datasets.py
    python download_datasets.py --dry-run
"""

import argparse, json, random, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import ANNOTATED_DIR, EMOJI_VOCAB_PATH, TRAIN_SPLIT, VAL_SPLIT

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: run  pip install 'datasets==2.18.0'"); sys.exit(1)

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

_AVIATION_KW = re.compile(
    r"\b(airline|airways|flight|airport|boeing|airbus|cancel|delay|"
    r"aircraft|landed|depart|terminal|runway|pilot|aviation|plane)\b", re.I
)
_ISSUE_KW = re.compile(
    r"\b(order|payment|refund|cancel|transfer|charge|invoice|failed|"
    r"declined|wrong|missing|damaged|broken|return|delivery|tracking)\b", re.I
)


def _get_names(ds_split, field):
    feat = ds_split.features.get(field)
    if feat is None: return []
    sub = getattr(feat, "feature", feat)
    return list(getattr(sub, "names", []))


def _map(tag_id, names, tag_map):
    raw = names[tag_id] if names and 0 <= tag_id < len(names) else "O"
    return tag_map.get(raw, "O")


# ═══════════════════════════════════════════════════════════════
# 1. CoNLL-2003  (aviation rows only)
# ═══════════════════════════════════════════════════════════════
CONLL_MAP = {
    "B-ORG":  "B-FLIGHT_ID", "I-ORG":  "I-FLIGHT_ID",
    "B-MISC": "B-EVENT",     "I-MISC": "I-EVENT",
    "B-LOC":  "B-FLIGHT_ID", "I-LOC":  "I-FLIGHT_ID",
}

def download_conll2003(dry_run=False):
    print("\n[1/4] CoNLL-2003 (aviation rows only) …")
    try:
        ds = load_dataset("conll2003", trust_remote_code=True)
    except Exception as e:
        print(f"    ✗ {e}"); return []

    samples = []
    for split in ["train", "validation", "test"]:
        sp = ds.get(split)
        if sp is None: continue
        names = _get_names(sp, "ner_tags")
        for row in sp:
            toks = list(row.get("tokens", []))
            text = " ".join(toks)
            if not _AVIATION_KW.search(text): continue
            labs = [_map(n, names, CONLL_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                samples.append({"text": text, "tokens": toks, "labels": labs,
                                 "domain": "aviation", "meta": {"source": "conll2003"}})
    print(f"    → {len(samples)} samples")
    if not dry_run:
        with open(RAW_DIR/"conll2003.jsonl","w") as f:
            for s in samples: f.write(json.dumps(s)+"\n")
    return samples


# ═══════════════════════════════════════════════════════════════
# 2. WNUT-17
# ═══════════════════════════════════════════════════════════════
WNUT_MAP = {
    "B-corporation": "B-FLIGHT_ID", "I-corporation": "I-FLIGHT_ID",
    "B-product":     "B-ORDER_ID",  "I-product":     "I-ORDER_ID",
    "B-group":       "B-FLIGHT_ID", "I-group":       "I-FLIGHT_ID",
    "B-location":    "B-FLIGHT_ID", "I-location":    "I-FLIGHT_ID",
}

def download_wnut17(dry_run=False):
    print("\n[2/4] WNUT-17 …")
    try:
        ds = load_dataset("wnut_17", trust_remote_code=True)
    except Exception as e:
        print(f"    ✗ {e}"); return []

    samples = []
    for split in ["train", "test", "validation"]:
        sp = ds.get(split)
        if sp is None: continue
        names = _get_names(sp, "ner_tags")
        for row in sp:
            toks = list(row.get("tokens", []))
            labs = [_map(n, names, WNUT_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                samples.append({"text": " ".join(toks), "tokens": toks, "labels": labs,
                                 "domain": "general", "meta": {"source": "wnut17"}})
    print(f"    → {len(samples)} samples")
    if not dry_run:
        with open(RAW_DIR/"wnut17.jsonl","w") as f:
            for s in samples: f.write(json.dumps(s)+"\n")
    return samples


# ═══════════════════════════════════════════════════════════════
# 3. Banking77  (real text only)
# ═══════════════════════════════════════════════════════════════
def download_banking77(dry_run=False):
    print("\n[3/4] Banking77 (real text, issue-keyword filter) …")
    try:
        ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset("banking77", trust_remote_code=True)
        except Exception as e:
            print(f"    ✗ {e}"); return []

    id2label = {str(i): n for i, n in enumerate(_get_names(ds["train"], "label"))}
    samples = []
    for split in ["train", "test"]:
        sp = ds.get(split)
        if sp is None: continue
        for row in sp:
            text   = str(row.get("text", "")).strip()
            intent = id2label.get(str(row.get("label", "")), "")
            if not text or not _ISSUE_KW.search(text): continue
            toks = text.split()
            labs = ["O"] * len(toks)
            for i, t in enumerate(toks):
                if re.fullmatch(r"#?\d{4,}", re.sub(r"[.,!?;:()\[\]]","",t)):
                    labs[i] = "B-ORDER_ID"
            if all(l == "O" for l in labs): continue
            samples.append({"text": text, "tokens": toks, "labels": labs,
                             "domain": "ecommerce",
                             "meta": {"source": "banking77", "intent": intent}})
    print(f"    → {len(samples)} samples")
    if not dry_run:
        with open(RAW_DIR/"banking77.jsonl","w") as f:
            for s in samples: f.write(json.dumps(s)+"\n")
    return samples


# ═══════════════════════════════════════════════════════════════
# 4. CoNLL++ (higher quality relabeled conll)
# ═══════════════════════════════════════════════════════════════
CONLLPP_MAP = {
    "B-ORG":  "B-FLIGHT_ID", "I-ORG":  "I-FLIGHT_ID",
    "B-MISC": "B-EVENT",     "I-MISC": "I-EVENT",
}

def download_conllpp(dry_run=False):
    print("\n[4/4] CoNLL++ (higher-quality NER, aviation rows) …")
    try:
        ds = load_dataset("conllpp", trust_remote_code=True)
    except Exception as e:
        print(f"    ✗ {e}"); return []

    samples = []
    for split in ["train", "validation", "test"]:
        sp = ds.get(split)
        if sp is None: continue
        names = _get_names(sp, "ner_tags")
        for row in sp:
            toks = list(row.get("tokens", []))
            text = " ".join(toks)
            if not _AVIATION_KW.search(text): continue
            labs = [_map(n, names, CONLLPP_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                samples.append({"text": text, "tokens": toks, "labels": labs,
                                 "domain": "aviation", "meta": {"source": "conllpp"}})
    print(f"    → {len(samples)} samples")
    if not dry_run:
        with open(RAW_DIR/"conllpp.jsonl","w") as f:
            for s in samples: f.write(json.dumps(s)+"\n")
    return samples


# ═══════════════════════════════════════════════════════════════
# Merge & re-split
# ═══════════════════════════════════════════════════════════════
def merge_and_split(real_samples, dry_run=False):
    print(f"\n{'═'*60}")
    src_counts = Counter(s["meta"]["source"] for s in real_samples)
    print(f"  Total real samples: {len(real_samples):,}")
    for src, cnt in src_counts.most_common():
        print(f"    {src:<25} {cnt:>5}")

    random.shuffle(real_samples)
    n  = len(real_samples)
    nt = int(n * TRAIN_SPLIT)
    nv = int(n * VAL_SPLIT)

    splits = {"train": real_samples[:nt],
              "val":   real_samples[nt:nt+nv],
              "test":  real_samples[nt+nv:]}

    if dry_run:
        for k, v in splits.items(): print(f"  [dry-run] {k}: {len(v)}")
        return

    for name, data in splits.items():
        out = ANNOTATED_DIR / f"{name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in data: f.write(json.dumps(s, ensure_ascii=False)+"\n")
        print(f"  → {out}  ({len(data)} samples)")

    try:
        from preprocessing import build_vocab_from_texts, save_vocab
        vocab = build_vocab_from_texts([s["text"] for s in real_samples])
        save_vocab(vocab, EMOJI_VOCAB_PATH)
        print(f"  → Emoji vocab: {len(vocab)} entries")
    except Exception as e:
        print(f"  ⚠ Emoji vocab: {e}")

    print(f"\n  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-conll",   action="store_true")
    p.add_argument("--skip-wnut",    action="store_true")
    p.add_argument("--skip-banking", action="store_true")
    p.add_argument("--skip-conllpp", action="store_true")
    p.add_argument("--skip-merge",   action="store_true")
    p.add_argument("--dry-run",      action="store_true")
    args = p.parse_args()

    print("\n" + "═"*60)
    print("  MESSI — Real Dataset Downloader (ZERO synthetic data)")
    print("  CoNLL-2003 · WNUT-17 · Banking77 · CoNLL++")
    print("═"*60)

    all_real = []
    if not args.skip_conll:   all_real += download_conll2003(args.dry_run)
    if not args.skip_wnut:    all_real += download_wnut17(args.dry_run)
    if not args.skip_banking: all_real += download_banking77(args.dry_run)
    if not args.skip_conllpp: all_real += download_conllpp(args.dry_run)

    if not args.skip_merge:
        merge_and_split(all_real, dry_run=args.dry_run)
        if not args.dry_run:
            print("\n✅ Done! Run:  python train.py --epochs 60")


if __name__ == "__main__":
    main()
