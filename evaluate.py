"""
MESSI — Evaluation v2 (span-level seqeval + per-class breakdown + Cohen's κ)
=============================================================================
Usage:
    python evaluate.py --test data/annotated/test.jsonl
    python evaluate.py --test data/annotated/test.jsonl --baseline spacy
    python evaluate.py --test data/annotated/test.jsonl --baseline bert
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import cohen_kappa_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

from config import BEST_MODEL_PATH, IDX2TAG, TAG2IDX, NUM_TAGS
from preprocessing import build_emoji_aware_nlp, load_embedding_components
from model import BiLSTMCRF, NERDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test",     type=Path, default=Path("data/annotated/test.jsonl"))
    p.add_argument("--device",   default="auto")
    p.add_argument("--baseline", choices=["spacy","bert"], default=None)
    p.add_argument("--show-errors", action="store_true", help="Print FP/FN examples")
    return p.parse_args()


def select_device(d):
    if d == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def _spans_from_bio(tags):
    """Convert tag list → set of (label, start, end) spans."""
    spans = set()
    i = 0
    while i < len(tags):
        if tags[i].startswith("B-"):
            lbl = tags[i][2:]
            j   = i + 1
            while j < len(tags) and tags[j] == f"I-{lbl}":
                j += 1
            spans.add((lbl, i, j))
            i = j
        else:
            i += 1
    return spans


def span_level_f1(all_gold_seqs, all_pred_seqs):
    """Compute entity-level P/R/F1 (span-based, ignoring I- boundaries strictly)."""
    TP = FP = FN = 0
    per_label = {}
    for gold_seq, pred_seq in zip(all_gold_seqs, all_pred_seqs):
        gold_tags = [IDX2TAG.get(g, "O") for g in gold_seq]
        pred_tags = [IDX2TAG.get(p, "O") for p in pred_seq]
        gs = _spans_from_bio(gold_tags)
        ps = _spans_from_bio(pred_tags)
        for sp in ps:
            if sp in gs: TP += 1
            else:         FP += 1
            per_label.setdefault(sp[0], {"TP":0,"FP":0,"FN":0})["TP" if sp in gs else "FP"] += 1
        for sp in gs:
            if sp not in ps:
                FN += 1
                per_label.setdefault(sp[0], {"TP":0,"FP":0,"FN":0})["FN"] += 1

    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1, per_label


def eval_messi(args, device):
    ckpt = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    use_char = ckpt.get("use_char_cnn", False)
    model = BiLSTMCRF(emoji_vocab=ckpt["emoji_vocab"], use_char_cnn=use_char)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"[Eval] Checkpoint val_f1={ckpt.get('val_f1','?')}")

    nlp = build_emoji_aware_nlp()
    _, extractor = load_embedding_components(nlp)
    loader = DataLoader(
        NERDataset(args.test, extractor),
        batch_size=32, shuffle=False, collate_fn=collate_fn,
    )

    all_preds_flat, all_gold_flat   = [], []
    all_pred_seqs,  all_gold_seqs   = [], []
    entity_labels = [i for i in range(NUM_TAGS) if IDX2TAG.get(i,"O") != "O"]

    with torch.no_grad():
        for sv, ei, la, mask, lens, ch in loader:
            sv, ei, mask, ch = sv.to(device), ei.to(device), mask.to(device), ch.to(device)
            seqs = model.decode(sv, ei, mask, lens, char_ids=ch)
            for b, seq in enumerate(seqs):
                L = lens[b].item()
                ps = (seq + [TAG2IDX["O"]]*L)[:L]
                gs = la[b, :L].tolist()
                all_preds_flat.extend(ps)
                all_gold_flat.extend(gs)
                all_pred_seqs.append(ps)
                all_gold_seqs.append(gs)

    # Token-level report
    print("\n" + "="*62 + "\nMESSI BiLSTM-CRF — Token-Level Report\n" + "="*62)
    from sklearn.metrics import classification_report as creport
    print(creport(all_gold_flat, all_preds_flat,
                  labels=entity_labels,
                  target_names=[IDX2TAG[i] for i in entity_labels],
                  zero_division=0))
    macro_f1 = f1_score(all_gold_flat, all_preds_flat,
                        labels=entity_labels, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(all_gold_flat, all_preds_flat)
    print(f"Macro Token-F1 : {macro_f1:.4f}")
    print(f"Cohen's κ      : {kappa:.4f}")

    # Span-level (entity-level) report
    prec, rec, f1, per_label = span_level_f1(all_gold_seqs, all_pred_seqs)
    print("\n" + "="*62 + "\nSpan-Level (Entity) Report\n" + "="*62)
    print(f"{'Entity Type':<20} {'P':>7} {'R':>7} {'F1':>7}")
    print("-"*42)
    for lbl, counts in sorted(per_label.items()):
        tp  = counts["TP"]; fpp = counts["FP"]; fn = counts["FN"]
        p   = tp / max(tp + fpp, 1)
        r   = tp / max(tp + fn, 1)
        f   = 2*p*r / max(p+r, 1e-9)
        print(f"{lbl:<20} {p:>7.4f} {r:>7.4f} {f:>7.4f}")
    print("-"*42)
    print(f"{'MACRO (span)':<20} {prec:>7.4f} {rec:>7.4f} {f1:>7.4f}")

    if args.show_errors:
        _show_errors(all_gold_seqs, all_pred_seqs)


def _show_errors(gold_seqs, pred_seqs, n=10):
    print("\n[Error Analysis] First mismatch examples:")
    shown = 0
    for g, p in zip(gold_seqs, pred_seqs):
        if g != p and shown < n:
            gt = [IDX2TAG.get(i,"O") for i in g]
            pt = [IDX2TAG.get(i,"O") for i in p]
            print(f"  GOLD: {gt}")
            print(f"  PRED: {pt}")
            shown += 1


def main():
    args   = parse_args()
    device = select_device(args.device)
    if args.baseline == "spacy":
        from baselines import run_spacy_baseline
        run_spacy_baseline(args.test)
    elif args.baseline == "bert":
        from baselines import evaluate_bert
        evaluate_bert(args.test, device)
    else:
        eval_messi(args, device)


if __name__ == "__main__":
    main()
