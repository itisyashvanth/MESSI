"""
MESSI — Baseline Models
========================
Merges: spacy_ner_baseline.py + bert_ner_baseline.py

Run:
    python baselines.py --model spacy --test data/annotated/test.jsonl
    python baselines.py --model bert  --train data/annotated/train.jsonl \
                                      --val   data/annotated/val.jsonl
    python baselines.py --model bert  --test data/annotated/test.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, str(Path(__file__).parent))
from config import TAG2IDX, IDX2TAG, NUM_TAGS

BERT_MODEL_NAME  = "bert-base-uncased"
BERT_CKPT_PATH   = Path("models/bert_ner_best.pt")


# ═══════════════════════════════════════════════════════════════
#  spaCy Baseline
# ═══════════════════════════════════════════════════════════════

def run_spacy_baseline(test_path: Path):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    all_preds, all_gold = [], []

    with open(test_path) as f:
        for line in f:
            s = json.loads(line.strip())
            tokens, gold_strs = s["tokens"], s["labels"]
            gold_ids = [TAG2IDX.get(l, TAG2IDX["O"]) for l in gold_strs]

            text = " ".join(tokens)
            doc  = nlp(text)
            pred_ids = [TAG2IDX["O"]] * len(tokens)

            char_pos, c2t = 0, {}
            for i, t in enumerate(tokens):
                for c in range(char_pos, char_pos + len(t)):
                    c2t[c] = i
                char_pos += len(t) + 1

            for ent in doc.ents:
                i = c2t.get(ent.start_char)
                if i is None:
                    continue
                b = f"B-{ent.label_}"; iv = f"I-{ent.label_}"
                if b in TAG2IDX:
                    pred_ids[i] = TAG2IDX[b]
                    j = i + 1
                    while j < len(tokens):
                        cs = sum(len(tokens[k]) + 1 for k in range(j))
                        if cs >= ent.end_char:
                            break
                        if iv in TAG2IDX:
                            pred_ids[j] = TAG2IDX[iv]
                        j += 1

            all_preds.extend(pred_ids)
            all_gold.extend(gold_ids)

    entity_labels = [i for i in range(NUM_TAGS) if IDX2TAG.get(i, "O") != "O"]
    macro_f1 = f1_score(all_gold, all_preds, labels=entity_labels, average="macro", zero_division=0)
    print("\n" + "="*60)
    print("Baseline: Vanilla spaCy NER")
    print("="*60)
    print(classification_report(all_gold, all_preds,
                                 labels=entity_labels,
                                 target_names=[IDX2TAG[i] for i in entity_labels],
                                 zero_division=0))
    print(f"Macro F1: {macro_f1:.4f}")
    return macro_f1


# ═══════════════════════════════════════════════════════════════
#  BERT Baseline
# ═══════════════════════════════════════════════════════════════

class _BERTDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_len=128):
        self.samples, self.tokenizer, self.max_len = [], tokenizer, max_len
        with open(path) as f:
            for line in f:
                s = json.loads(line.strip())
                if s.get("tokens"):
                    self.samples.append(s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tokens, labels = s["tokens"], [TAG2IDX.get(l, TAG2IDX["O"]) for l in s["labels"]]
        enc = self.tokenizer(tokens, is_split_into_words=True, truncation=True,
                              max_length=self.max_len, padding="max_length", return_tensors="pt")
        word_ids = enc.word_ids()
        aligned  = []
        prev_wid = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev_wid:
                aligned.append(labels[wid] if wid < len(labels) else TAG2IDX["O"])
            else:
                aligned.append(-100)
            prev_wid = wid
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(aligned, dtype=torch.long),
        }


def _eval_bert(model, loader, device):
    model.eval()
    all_preds, all_gold = [], []
    entity_labels = [i for i in range(NUM_TAGS) if IDX2TAG.get(i, "O") != "O"]
    from torch.utils.data import DataLoader
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.argmax(-1).cpu().tolist()
            gold = batch["labels"].tolist()
            for p_seq, g_seq in zip(logits, gold):
                for p, g in zip(p_seq, g_seq):
                    if g != -100:
                        all_preds.append(p); all_gold.append(g)
    return f1_score(all_gold, all_preds, labels=entity_labels, average="macro", zero_division=0)


def train_bert(train_path, val_path, epochs=5, lr=2e-5, batch_size=16, device=None):
    from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_TAGS).to(device)
    train_dl = DataLoader(_BERTDataset(train_path, tokenizer), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(_BERTDataset(val_path,   tokenizer), batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sched     = get_linear_schedule_with_warmup(optimizer, len(train_dl)//10, len(train_dl)*epochs)
    best_f1   = 0.0

    for ep in range(1, epochs + 1):
        model.train(); total = 0
        for batch in train_dl:
            optimizer.zero_grad()
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); sched.step(); total += loss.item()
        f1 = _eval_bert(model, val_dl, device)
        print(f"[BERT Epoch {ep}/{epochs}] loss={total/len(train_dl):.4f}  val_f1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "val_f1": f1}, BERT_CKPT_PATH)
            print(f"  ✓ Saved (f1={f1:.4f})")
    return best_f1


def evaluate_bert(test_path, device=None):
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    from torch.utils.data import DataLoader

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL_NAME, num_labels=NUM_TAGS)
    if BERT_CKPT_PATH.exists():
        model.load_state_dict(torch.load(BERT_CKPT_PATH, map_location=device)["model_state"])
    model = model.to(device)
    test_dl = DataLoader(_BERTDataset(test_path, tokenizer), batch_size=16)
    f1 = _eval_bert(model, test_dl, device)
    print(f"\n[BERT Baseline] Macro F1 = {f1:.4f}")
    return f1


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  choices=["spacy", "bert"], required=True)
    p.add_argument("--train",  type=Path, default=None)
    p.add_argument("--val",    type=Path, default=None)
    p.add_argument("--test",   type=Path, default=None)
    p.add_argument("--epochs", type=int,  default=5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "spacy":
        run_spacy_baseline(args.test)
    else:
        if args.train and args.val:
            train_bert(args.train, args.val, epochs=args.epochs, device=device)
        if args.test:
            evaluate_bert(args.test, device)


if __name__ == "__main__":
    main()
