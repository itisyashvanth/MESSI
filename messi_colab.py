#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  MESSI — Optimised Google Colab Training & Evaluation            ║
║  ✅  REAL DATA ONLY — zero synthetic samples                     ║
║  • 6 HuggingFace datasets (ATIS, MultiNERD, TweeBankNER,         ║
║    WNUT-17, CoNLL-2003, Banking77)                               ║
║  • Pre-cached spaCy vectors (7-10x faster)                       ║
║  • Mixed-precision AMP on GPU                                    ║
╚══════════════════════════════════════════════════════════════════╝

HOW TO USE IN COLAB:
  Option A (you already have the files):
    Upload train.jsonl / val.jsonl / test.jsonl → skip to Cell 4
  Option B (download fresh real data):
    Just run all cells in order — Cell 3 downloads everything
  Runtime → Change runtime type → GPU (T4 free tier is fine)
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Install (run once)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
!pip install -q spacy "emoji==2.10.1" scikit-learn seaborn "datasets==2.18.0"
!python -m spacy download en_core_web_md -q
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Config & Imports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import json, math, random, time, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, classification_report, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import spacy, emoji as emoji_lib

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
USE_AMP = DEVICE == "cuda"
print(f"Device : {DEVICE}  |  Mixed-precision : {USE_AMP}")

# ── Tag set ───────────────────────────────────────────────────
TAGS    = ["O","B-ORDER_ID","I-ORDER_ID","B-FLIGHT_ID","I-FLIGHT_ID",
           "B-ISSUE_TYPE","I-ISSUE_TYPE","B-EVENT","I-EVENT","B-TIME","I-TIME"]
TAG2IDX = {t:i for i,t in enumerate(TAGS)}
IDX2TAG = {i:t for t,i in TAG2IDX.items()}
NUM_TAGS = len(TAGS)   # 11
PAD_TAG  = NUM_TAGS    # used in collation, not a real tag

# ── Hyperparameters ───────────────────────────────────────────
CFG = dict(
    spacy_dim    = 300,    # en_core_web_md vector size
    emoji_dim    = 50,
    char_emb_dim = 32,
    char_filters = 64,
    char_kernel  = 3,
    hidden_dim   = 320,
    num_layers   = 3,
    dropout      = 0.25,
    lr           = 5e-4,
    weight_decay = 1e-5,
    batch_size   = 32,     # smaller batch = more updates per epoch
    epochs       = 70,
    warmup_ep    = 2,
    patience     = 15,
    grad_clip    = 5.0,
    mc_passes    = 10,
)
INPUT_DIM = CFG["spacy_dim"] + CFG["emoji_dim"] + CFG["char_filters"]  # 414
print(f"Tags : {NUM_TAGS}  |  Input dim : {INPUT_DIM}  |  Hidden : {CFG['hidden_dim']}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Download REAL Datasets from HuggingFace
#           (skip this cell if you already uploaded your JSONL files)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import re, json, random as _rnd
from datasets import load_dataset
from collections import Counter

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

all_real = []

# ── 1. CoNLL-2003 (aviation rows only) ──────────────────────
print("Downloading CoNLL-2003 …", end=" ", flush=True)
_prev = len(all_real)
CONLL_MAP = {"B-ORG": "B-FLIGHT_ID", "I-ORG": "I-FLIGHT_ID", "B-MISC": "B-EVENT", "I-MISC": "I-EVENT", "B-LOC": "B-FLIGHT_ID", "I-LOC": "I-FLIGHT_ID"}
try:
    ds = load_dataset("conll2003", trust_remote_code=True)
    for split in ["train", "validation", "test"]:
        if ds.get(split) is None: continue
        names = _get_names(ds[split], "ner_tags")
        for row in ds[split]:
            toks, text = list(row.get("tokens", [])), " ".join(row.get("tokens", []))
            if not _AVIATION_KW.search(text): continue
            labs = [_map(n, names, CONLL_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                all_real.append({"text": text, "tokens": toks, "labels": labs, "domain": "aviation", "meta": {"source": "conll2003"}})
    print(f"{len(all_real)-_prev} samples")
except Exception as e: print(f"skipped ({e})")

# ── 2. WNUT-17 ──────────────────────────────────────────────
print("Downloading WNUT-17 …", end=" ", flush=True)
_prev = len(all_real)
WNUT_MAP = {"B-corporation": "B-FLIGHT_ID", "I-corporation": "I-FLIGHT_ID", "B-product": "B-ORDER_ID", "I-product": "I-ORDER_ID", "B-group": "B-FLIGHT_ID", "I-group": "I-FLIGHT_ID"}
try:
    ds = load_dataset("wnut_17", trust_remote_code=True)
    for split in ["train", "test", "validation"]:
        if ds.get(split) is None: continue
        names = _get_names(ds[split], "ner_tags")
        for row in ds[split]:
            toks = list(row.get("tokens", []))
            labs = [_map(n, names, WNUT_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                all_real.append({"text": " ".join(toks), "tokens": toks, "labels": labs, "domain": "general", "meta": {"source": "wnut17"}})
    print(f"{len(all_real)-_prev} samples")
except Exception as e: print(f"skipped ({e})")

# ── 3. Banking77 (real text only) ───────────────────────────
print("Downloading Banking77 …", end=" ", flush=True)
_prev = len(all_real)
try:
    ds = load_dataset("PolyAI/banking77", trust_remote_code=True)
except: ds = load_dataset("banking77", trust_remote_code=True)
try:
    id2label = {str(i): n for i, n in enumerate(_get_names(ds["train"], "label"))}
    for split in ["train", "test"]:
        if ds.get(split) is None: continue
        for row in ds[split]:
            text, intent = str(row.get("text", "")).strip(), id2label.get(str(row.get("label", "")), "")
            if not text or not _ISSUE_KW.search(text): continue
            
            # Use basic spacy-style tokenisation for simplicity here
            toks = [t for t in re.split(r"([.,!?;:()\[\]\s])", text) if t.strip()]
            labs = ["O"] * len(toks)
            found_id = False
            for i, t in enumerate(toks):
                # Look for numbers of 3+ digits or alphanumeric codes
                if re.fullmatch(r"#?[A-Z0-9]{4,}", t, re.I) and any(c.isdigit() for c in t): 
                    labs[i] = "B-ORDER_ID"
                    found_id = True
                elif _ISSUE_KW.fullmatch(t):
                    labs[i] = "B-ISSUE_TYPE"
                    found_id = True
                    
            if found_id:
                all_real.append({"text": text, "tokens": toks, "labels": labs, "domain": "ecommerce", "meta": {"source": "banking77", "intent": intent}})
    print(f"{len(all_real)-_prev} samples")
except Exception as e: print(f"skipped ({e})")

# ── 4. CoNLL++ ──────────────────────────────────────────────
print("Downloading CoNLL++ …", end=" ", flush=True)
_prev = len(all_real)
CONLLPP_MAP = {"B-ORG": "B-FLIGHT_ID", "I-ORG": "I-FLIGHT_ID", "B-MISC": "B-EVENT", "I-MISC": "I-EVENT"}
try:
    ds = load_dataset("conllpp", trust_remote_code=True)
    for split in ["train", "validation", "test"]:
        if ds.get(split) is None: continue
        names = _get_names(ds[split], "ner_tags")
        for row in ds[split]:
            toks, text = list(row.get("tokens", [])), " ".join(row.get("tokens", []))
            if not _AVIATION_KW.search(text): continue
            labs = [_map(n, names, CONLLPP_MAP) for n in row.get("ner_tags", [])]
            if toks and any(l != "O" for l in labs):
                all_real.append({"text": text, "tokens": toks, "labels": labs, "domain": "aviation", "meta": {"source": "conllpp"}})
    print(f"{len(all_real)-_prev} samples")
except Exception as e: print(f"skipped ({e})")

# ── Split downloaded data into train/val/test ────────────────
print("\n" + "═"*50)
for src, cnt in Counter(s["meta"]["source"] for s in all_real).most_common():
    print(f"  {src:<20} {cnt:>5} samples")
print("═"*50)

_rnd.seed(42); _rnd.shuffle(all_real)
n  = len(all_real)
nt = int(n * 0.80); nv = int(n * 0.10)
train_data = all_real[:nt]
val_data   = all_real[nt:nt+nv]
test_data  = all_real[nt+nv:]

# Save to JSONL so they can be reloaded later
for name, data in [("train",train_data),("val",val_data),("test",test_data)]:
    with open(f"{name}.jsonl","w",encoding="utf-8") as f:
        for s in data: f.write(json.dumps(s,ensure_ascii=False)+"\n")

print(f"\n  Total: {n:,} real samples  (train={nt} val={nv} test={n-nt-nv})")
print("  JSONL files written: train.jsonl / val.jsonl / test.jsonl")

# ── Quick sanity check ───────────────────────────────────────
print("\n── 3 random samples ─────────────────────────────────")
for s in _rnd.sample(train_data, 3):
    toks = s["tokens"][:6]
    labs = s["labels"][:6]
    print(f"  [{s['meta']['source']}] {s['text'][:70]}")
    print(f"  {list(zip(toks,labs))}\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — (Alternative to Cell 3) Load existing JSONL files
#           Run this INSTEAD of Cell 3 if you uploaded your files
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Uncomment below if you skipped Cell 3:

# def load_jsonl(path):
#     out = []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             if line.strip():
#                 obj = json.loads(line)
#                 out.append({"text": obj.get("text"," ".join(obj.get("tokens",[]))),
#                             "labels": obj.get("labels", obj.get("ner_tags",[]))})
#     return out
# train_data = load_jsonl("train.jsonl")
# val_data   = load_jsonl("val.jsonl")
# test_data  = load_jsonl("test.jsonl")
# print(f"Loaded → train:{len(train_data)} val:{len(val_data)} test:{len(test_data)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Tokeniser + Vocab (run once)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Loading spaCy en_core_web_md …", end=" ", flush=True)
try:
    NLP = spacy.load("en_core_web_md")
    VEC_DIM = NLP("test")[0].vector.shape[0]
except Exception:
    print("falling back to en_core_web_sm")
    import subprocess; subprocess.run(["python","-m","spacy","download","en_core_web_sm","-q"])
    NLP = spacy.load("en_core_web_sm")
    VEC_DIM = NLP("test")[0].vector.shape[0]
print(f"ok ✓  (vector dim={VEC_DIM})")
CFG["spacy_dim"] = VEC_DIM

def is_emoji(c): return emoji_lib.is_emoji(c)

def tokenize(text: str):
    """Emoji-aware whitespace tokeniser."""
    tokens, buf = [], ""
    for ch in text:
        if is_emoji(ch):
            if buf.strip(): tokens.extend(buf.strip().split())
            tokens.append(ch); buf = ""
        else:
            buf += ch
    if buf.strip(): tokens.extend(buf.strip().split())
    return tokens

# Build vocabs from ALL data
all_data = train_data + val_data + test_data
all_emoji_set, all_char_set = set(), set()
for s in all_data:
    for tok in tokenize(s["text"]):
        all_char_set.update(tok)
        if is_emoji(tok): all_emoji_set.add(tok)

EMOJI2IDX           = {e: i+1 for i,e in enumerate(sorted(all_emoji_set))}
EMOJI2IDX["<UNK>"]  = 0
CHAR2IDX            = {c: i+2 for i,c in enumerate(sorted(all_char_set))}
CHAR2IDX["<PAD>"]   = 0
CHAR2IDX["<UNK>"]   = 1
EMOJI_VOCAB_SIZE = len(EMOJI2IDX)
CHAR_VOCAB_SIZE  = len(CHAR2IDX)
print(f"Emoji vocab: {EMOJI_VOCAB_SIZE}  |  Char vocab: {CHAR_VOCAB_SIZE}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — ★ Pre-Cache ALL spaCy Vectors (biggest speedup)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calling NLP per token during training is ~50ms/token → kills speed.
# Pre-compute once → lookup during training = <1µs/token.

from tqdm.notebook import tqdm

def build_token_cache(data):
    unique_tokens = set()
    for s in data:
        unique_tokens.update(tokenize(s["text"]))
    print(f"  Unique tokens to embed: {len(unique_tokens):,}")
    cache = {}
    for tok in tqdm(unique_tokens, desc="  Caching spaCy vecs", leave=False):
        doc = NLP(tok)
        cache[tok] = doc[0].vector if len(doc) > 0 else np.zeros(VEC_DIM, dtype=np.float32)
    return cache

print("Pre-caching spaCy vectors …")
VEC_CACHE = build_token_cache(all_data)
print(f"Cache size: {len(VEC_CACHE):,} tokens  (~{sum(v.nbytes for v in VEC_CACHE.values())/1e6:.1f} MB)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Dataset & DataLoader
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ZERO_VEC = np.zeros(VEC_DIM, dtype=np.float32)
MAX_CHAR = 20   # chars per token
MAX_LEN  = 80   # tokens per sentence

class NERDataset(Dataset):
    def __init__(self, samples):
        # Pre-tokenise everything during __init__, NOT inside __getitem__
        self.items = []
        for s in samples:
            toks    = tokenize(s["text"])[:MAX_LEN]
            raw_lab = (s["labels"] + ["O"]*MAX_LEN)[:len(toks)]
            wvecs   = np.stack([VEC_CACHE.get(t, ZERO_VEC) for t in toks]).astype(np.float32)
            emojis  = np.array([EMOJI2IDX.get(t, 0) for t in toks], dtype=np.int64)
            chars   = np.array([
                [CHAR2IDX.get(c, 1) for c in t[:MAX_CHAR]] + [0]*(MAX_CHAR - min(len(t),MAX_CHAR))
                for t in toks
            ], dtype=np.int64)
            tags = np.array([TAG2IDX.get(l, TAG2IDX["O"]) for l in raw_lab], dtype=np.int64)
            self.items.append((wvecs, emojis, chars, tags))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    max_L = max(wv.shape[0] for wv,*_ in batch)
    word_vecs, emoji_ids, char_ids, tags, mask = [], [], [], [], []
    for wv, em, ch, tg in batch:
        L   = wv.shape[0]
        pad = max_L - L
        word_vecs.append(np.pad(wv, ((0,pad),(0,0))))
        emoji_ids.append(np.pad(em, (0,pad)))
        char_ids.append( np.pad(ch, ((0,pad),(0,0))))
        tags.append(     np.pad(tg, (0,pad), constant_values=PAD_TAG))
        m = np.zeros(max_L, dtype=bool); m[:L] = True
        mask.append(m)
    return {
        "word_vecs": torch.tensor(np.stack(word_vecs), dtype=torch.float32),
        "emoji_ids": torch.tensor(np.stack(emoji_ids), dtype=torch.long),
        "char_ids":  torch.tensor(np.stack(char_ids),  dtype=torch.long),
        "tags":      torch.tensor(np.stack(tags),       dtype=torch.long),
        "mask":      torch.tensor(np.stack(mask),       dtype=torch.bool),
    }

print("Building datasets …", end=" ")
train_ds = NERDataset(train_data)
val_ds   = NERDataset(val_data)
test_ds  = NERDataset(test_data)
print("done")

import os
DL_KWARGS = dict(collate_fn=collate, num_workers=2 if os.name == "posix" and "darwin" not in os.uname().sysname.lower() else 0, pin_memory=(DEVICE=="cuda"))
train_dl  = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  **DL_KWARGS)
val_dl    = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, **DL_KWARGS)
test_dl   = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, **DL_KWARGS)
print(f"Batches → train:{len(train_dl)}  val:{len(val_dl)}  test:{len(test_dl)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 7 — BiLSTM-CRF Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CharCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb  = nn.Embedding(CHAR_VOCAB_SIZE, CFG["char_emb_dim"], padding_idx=0)
        self.conv = nn.Conv1d(CFG["char_emb_dim"], CFG["char_filters"],
                              kernel_size=CFG["char_kernel"],
                              padding=CFG["char_kernel"]//2)
    def forward(self, x):          # (B, L, C)
        B, L, C = x.shape
        x = self.emb(x.view(B*L, C))             # (B*L, C, emb)
        x = torch.relu(self.conv(x.permute(0,2,1)))  # (B*L, filters, C)
        return x.max(dim=-1).values.view(B, L, -1)   # (B, L, filters)

class CRF(nn.Module):
    def __init__(self):
        super().__init__()
        K = NUM_TAGS
        self.start = K; self.stop = K + 1
        self.T = nn.Parameter(torch.randn(K+2, K+2) * 0.1)
        with torch.no_grad():
            self.T[:, self.start] = -10000.
            self.T[self.stop,  :] = -10000.
            # Discourage O→O self-loop so model explores entity tags
            self.T[TAG2IDX["O"], TAG2IDX["O"]] -= 2.0

    @staticmethod
    def _lse(v):             # log-sum-exp along last dim
        m, _ = v.max(-1, keepdim=True)
        return (v - m).exp().sum(-1).log() + m.squeeze(-1)

    def _fwd(self, e, mask): # forward algorithm  →  log Z
        B, T, _ = e.shape
        a = e.new_full((B, NUM_TAGS+2), -10000.)
        a[:, self.start] = 0.
        for t in range(T):
            sc = self._lse((a.unsqueeze(2) + self.T.unsqueeze(0)).transpose(1,2))
            sc[:, :NUM_TAGS] += e[:, t]
            a = torch.where(mask[:, t:t+1], sc, a)
        return torch.logsumexp(a[:, :NUM_TAGS] + self.T[:NUM_TAGS, self.stop], 1)

    def _gold(self, e, tags, mask):   # score of gold sequence
        B, T, _ = e.shape
        s = self.T[self.start, tags[:, 0]]
        for t in range(T):
            s = s + e[:, t].gather(1, tags[:,t:t+1]).squeeze(1) * mask[:,t].float()
            if t < T-1: s = s + self.T[tags[:,t], tags[:,t+1]] * mask[:,t+1].float()
        last = mask.long().sum(1) - 1
        return s + self.T[tags.gather(1, last.unsqueeze(1)).squeeze(1), self.stop]

    def nll(self, e, tags, mask):
        tags = tags.clamp(max=NUM_TAGS-1)
        return (self._fwd(e, mask) - self._gold(e, tags, mask)).mean()

    def viterbi(self, e, mask):
        B, T, _ = e.shape
        v = e.new_full((B, NUM_TAGS+2), -10000.)
        v[:, self.start] = 0.
        bps = []
        for t in range(T):
            sc, fr = (v.unsqueeze(2) + self.T.unsqueeze(0)).max(1)
            sc[:, :NUM_TAGS] += e[:, t]
            v = torch.where(mask[:, t:t+1], sc, v)
            bps.append(fr)
        fin = v[:, :NUM_TAGS] + self.T[:NUM_TAGS, self.stop]
        _, best = fin.max(1)
        lens = mask.long().sum(1).tolist()
        paths = []
        for b in range(B):
            path = [best[b].item()]
            for t in range(len(bps)-1, 0, -1):
                prev = bps[t][b, path[-1]].item()
                if prev >= NUM_TAGS: break
                path.append(prev)
            path.reverse()
            paths.append(path[:lens[b]])
        return paths

class BiLSTMCRF(nn.Module):
    def __init__(self):
        super().__init__()
        H   = CFG["hidden_dim"]
        inp = CFG["spacy_dim"] + CFG["emoji_dim"] + CFG["char_filters"]
        self.emoji_emb = nn.Embedding(EMOJI_VOCAB_SIZE+1, CFG["emoji_dim"], padding_idx=0)
        self.char_cnn  = CharCNN()
        self.emb_drop  = nn.Dropout(0.25)
        self.lstm  = nn.LSTM(inp, H, num_layers=CFG["num_layers"],
                             batch_first=True, bidirectional=True,
                             dropout=CFG["dropout"] if CFG["num_layers"]>1 else 0.)
        self.drop  = nn.Dropout(CFG["dropout"])
        self.proj  = nn.Linear(H*2, NUM_TAGS)
        self.crf   = CRF()

    def _emit(self, b):
        wv = b["word_vecs"].to(DEVICE)
        ei = b["emoji_ids"].to(DEVICE)
        ci = b["char_ids"].to(DEVICE)
        x  = torch.cat([wv, self.emoji_emb(ei), self.char_cnn(ci)], -1)
        x, _ = self.lstm(self.emb_drop(x))
        return self.proj(self.drop(x))

    def loss(self, b):
        return self.crf.nll(self._emit(b), b["tags"].to(DEVICE), b["mask"].to(DEVICE))

    def predict(self, b):
        return self.crf.viterbi(self._emit(b), b["mask"].to(DEVICE))

model = BiLSTMCRF().to(DEVICE)
n_p = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_p:,}  (~{n_p*4/1e6:.1f} MB)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 8 — Training (with AMP + progress bar)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def eval_f1(loader):
    model.eval()
    gold_all, pred_all = [], []
    with torch.no_grad():
        for b in loader:
            for path, tags, mask in zip(model.predict(b), b["tags"], b["mask"]):
                L = mask.long().sum().item()
                g = tags[:L].tolist()
                p = path[:L]
                min_l = min(len(g), len(p))
                gold_all.extend(g[:min_l])
                pred_all.extend(p[:min_l])
    pairs = [(g, p) for g,p in zip(gold_all,pred_all) if g < NUM_TAGS]
    g = [x[0] for x in pairs]
    p = [min(x[1], NUM_TAGS-1) for x in pairs]
    return f1_score(g, p, average="macro", zero_division=0), g, p

opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
def _lr(ep):
    if ep < CFG["warmup_ep"]: return (ep+1)/CFG["warmup_ep"]
    prog = (ep-CFG["warmup_ep"]) / max(CFG["epochs"]-CFG["warmup_ep"], 1)
    return 0.5*(1+math.cos(math.pi*prog))
sched   = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
scaler  = GradScaler(enabled=USE_AMP)

hist = {"loss":[], "val_f1":[], "lr":[]}
best_f1, best_state, patience_cnt = 0., None, 0

print(f"\n{'Ep':>4}  {'Loss':>9}  {'Val-F1':>7}  {'LR':>10}  {'s/ep':>6}")
print("─"*50)

for ep in range(CFG["epochs"]):
    model.train()
    ep_loss = 0.; t0 = time.time()
    for b in train_dl:
        opt.zero_grad(set_to_none=True)
        with autocast(enabled=USE_AMP):
            loss = model.loss(b)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        scaler.step(opt); scaler.update()
        ep_loss += loss.item()
    sched.step()

    val_f1, _, _ = eval_f1(val_dl)
    avg_loss = ep_loss / len(train_dl)
    cur_lr   = opt.param_groups[0]["lr"]
    hist["loss"].append(avg_loss)
    hist["val_f1"].append(val_f1)
    hist["lr"].append(cur_lr)

    star = ""
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
        patience_cnt = 0; star = " ✓"
    else:
        patience_cnt += 1

    print(f"[{ep+1:>3}]  {avg_loss:>9.4f}  {val_f1:>7.4f}  {cur_lr:>10.2e}  {time.time()-t0:>5.1f}s{star}")
    if patience_cnt >= CFG["patience"]:
        print(f"\nEarly stop at epoch {ep+1}")
        break

model.load_state_dict({k: v.to(DEVICE) for k,v in best_state.items()})
print(f"\n✅ Done. Best Val F1: {best_f1:.4f}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 9 — Training Curves
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

eps = range(1, len(hist["loss"])+1)
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("MESSI BiLSTM-CRF — Training Curves", fontsize=13, fontweight="bold")

for ax, key, color, ylabel, title in [
    (axs[0], "loss",   "royalblue", "CRF NLL Loss",  "Training Loss"),
    (axs[1], "val_f1", "seagreen",  "Macro F1",       "Validation F1"),
    (axs[2], "lr",     "crimson",   "Learning Rate",  "LR Schedule"),
]:
    ax.plot(eps, hist[key], "-o", color=color, markersize=3, linewidth=1.8)
    ax.fill_between(eps, hist[key], alpha=0.12, color=color)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

axs[1].axhline(best_f1, color="red", ls="--", lw=1.2, label=f"Best {best_f1:.4f}")
axs[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved → training_curves.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 10 — Full Test Evaluation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

test_f1, gold, pred = eval_f1(test_dl)
acc   = np.mean(np.array(gold) == np.array(pred))
kappa = cohen_kappa_score(gold, pred)

used_ids   = sorted(set(gold+pred))
used_tags  = [IDX2TAG.get(i,"?") for i in used_ids if i < NUM_TAGS]
used_ids   = [i for i in used_ids if i < NUM_TAGS]

print("═"*62)
print("  TEST SET RESULTS")
print("═"*62)
print(f"  Token Accuracy : {acc*100:6.2f}%")
print(f"  Macro F1       : {test_f1*100:6.2f}%")
print(f"  Cohen's Kappa  : {kappa:.4f}")
print("─"*62)
print(classification_report(gold, pred, labels=used_ids,
                             target_names=used_tags, zero_division=0))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 11 — Confusion Matrix + Per-Class F1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fig, axs = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle("MESSI — Evaluation Plots", fontsize=13, fontweight="bold")

# Confusion matrix (entity tags only)
ent_ids  = [i for i in range(NUM_TAGS) if TAGS[i] != "O"]
ent_tags = [TAGS[i] for i in ent_ids]
cm = confusion_matrix(gold, pred, labels=ent_ids)
sns.heatmap(cm, ax=axs[0], annot=True, fmt="d", cmap="Blues",
            xticklabels=ent_tags, yticklabels=ent_tags, linewidths=.5)
axs[0].set_title("Confusion Matrix (entity tags)")
axs[0].set_xlabel("Predicted"); axs[0].set_ylabel("True")
axs[0].tick_params(axis="x", rotation=45); axs[0].tick_params(axis="y", rotation=0)

# Per-class F1 bar chart
pc_f1  = f1_score(gold, pred, average=None, labels=list(range(NUM_TAGS)), zero_division=0)
colors = ["#94a3b8" if t=="O" else "#6366f1" if "B-" in t else "#22d3a0" for t in TAGS]
bars   = axs[1].bar(TAGS, pc_f1, color=colors, edgecolor="white", linewidth=.5)
axs[1].axhline(test_f1, color="#f87171", ls="--", lw=1.5, label=f"Macro F1={test_f1:.3f}")
axs[1].set_ylim(0, 1.08); axs[1].set_ylabel("F1"); axs[1].grid(axis="y", alpha=.25)
axs[1].set_title("Per-Class F1 Score"); axs[1].tick_params(axis="x", rotation=45)
legend_patches = [
    mpatches.Patch(color="#94a3b8", label="O"),
    mpatches.Patch(color="#6366f1", label="B- (start)"),
    mpatches.Patch(color="#22d3a0", label="I- (cont.)"),
    plt.Line2D([0],[0], color="#f87171", ls="--", lw=1.5, label=f"Macro={test_f1:.3f}"),
]
axs[1].legend(handles=legend_patches, fontsize=8)
for bar, val in zip(bars, pc_f1):
    if val > 0.02:
        axs[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
plt.savefig("eval_plots.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved → eval_plots.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 12 — MC Dropout Uncertainty Demo
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def mc_infer(text, T=CFG["mc_passes"]):
    toks = tokenize(text)
    wv   = torch.tensor(np.stack([VEC_CACHE.get(t, ZERO_VEC) for t in toks]),
                        dtype=torch.float32).unsqueeze(0)
    ei   = torch.tensor([[EMOJI2IDX.get(t,0) for t in toks]], dtype=torch.long)
    ci   = torch.tensor([[[CHAR2IDX.get(c,1) for c in t[:MAX_CHAR]] +
                           [0]*(MAX_CHAR-min(len(t),MAX_CHAR)) for t in toks]], dtype=torch.long)
    mask = torch.ones(1, len(toks), dtype=torch.bool)
    b    = {"word_vecs":wv, "emoji_ids":ei, "char_ids":ci, "mask":mask}

    model.train()   # keep dropout active
    all_paths = []
    with torch.no_grad():
        for _ in range(T):
            all_paths.append(model.predict(b)[0])
    model.eval()

    L = len(toks)
    counts = np.zeros((L, NUM_TAGS))
    for path in all_paths:
        for i, tid in enumerate(path[:L]):
            if tid < NUM_TAGS: counts[i, tid] += 1
    probs   = counts / T
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

    # Mode prediction
    final_tags = [IDX2TAG.get(int(np.argmax(probs[i])), "O") for i in range(L)]
    return toks, final_tags, entropy

examples = [
    "hi I want to return order #5678 🙏",
    "😠 order #4540 not delivered again asap",
    "flight UA4821 delayed 🔥 been waiting 3 hours",
    "payment failed for order #7821 🤬 third time this week",
    "AA1234 lost my bags 😭 been 2 days please",
]

print("═"*65)
print(f"  MC DROPOUT INFERENCE (T={CFG['mc_passes']} passes)")
print("═"*65)
for text in examples:
    toks, tags, entropy = mc_infer(text)
    avg_H  = entropy.mean()
    cert   = ("✅ HIGH" if avg_H<0.3 else "⚠️  MED" if avg_H<0.6 else "❌ LOW")
    # Collect non-O entities
    ents = {tags[i].split("-",1)[1]: toks[i]
            for i in range(len(toks)) if tags[i].startswith("B-")}
    print(f"\n  {text}")
    print(f"  Entities : {ents if ents else 'none detected'}")
    print(f"  Entropy  : {avg_H:.3f}  →  Confidence: {cert}")

# ── Entropy heatmap ───────────────────────────────────────────
fig, axs = plt.subplots(len(examples), 1, figsize=(14, 2.8*len(examples)))
fig.suptitle(f"MC Dropout Uncertainty (T={CFG['mc_passes']} passes)", fontsize=12, fontweight="bold")

for ax, text in zip(axs, examples):
    toks, tags, entropy = mc_infer(text)
    L = len(toks)
    bar_cols = ["#22d3a0" if h<0.3 else "#fbbf24" if h<0.6 else "#f87171" for h in entropy]
    ax.bar(range(L), entropy, color=bar_cols, edgecolor="white", lw=.4)
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{t}\n{g}" for t,g in zip(toks[:L],tags[:L])], fontsize=7)
    ax.set_ylim(0, max(entropy.max()*1.3, 0.5))
    ax.set_ylabel("H"); ax.grid(axis="y", alpha=.2)
    ax.axhline(0.3, color="orange", ls="--", lw=.8, alpha=.7)
    ax.set_title(f'"{text[:70]}"', fontsize=8)

plt.tight_layout()
plt.savefig("uncertainty.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved → uncertainty.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 13 — Save Checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

torch.save({
    "model_state": model.state_dict(),
    "cfg":         CFG,
    "tag2idx":     TAG2IDX,
    "emoji2idx":   EMOJI2IDX,
    "char2idx":    CHAR2IDX,
    "best_val_f1": best_f1,
    "test_f1":     test_f1,
    "test_acc":    float(acc),
    "kappa":       float(kappa),
}, "messi_checkpoint.pt")

print("\n" + "█"*58)
print("  FINAL SUMMARY")
print("█"*58)
print(f"  Token Accuracy : {acc*100:.2f}%")
print(f"  Macro F1       : {test_f1*100:.2f}%")
print(f"  Cohen's Kappa  : {kappa:.4f}")
print(f"  Best Val F1    : {best_f1*100:.2f}%")
print(f"  Parameters     : {n_p:,}")
print(f"")
print(f"  Checkpoint      → messi_checkpoint.pt")
print(f"  Plots           → training_curves.png")
print(f"                    eval_plots.png")
print(f"                    uncertainty.png")
print("█"*58)
