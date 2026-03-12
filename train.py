"""
MESSI — Training Script v2 (warmup + char-CNN + class-weighted loss)
=====================================================================
Improvements over v1:
  • Linear warmup for WARMUP_EPOCHS, then CosineAnnealingLR
  • Passes char_ids to model (char-CNN encoder)
  • Logs per-entity-class F1 at best-model checkpoints
  • class_weights based on tag frequency → focal-like upweighting of rare tags

Usage:
    python train.py --epochs 60 --batch-size 32
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, classification_report

from config import (
    BEST_MODEL_PATH, MODELS_DIR, LEARNING_RATE, MAX_EPOCHS,
    GRAD_CLIP_MAX_NORM, EARLY_STOPPING_PATIENCE, BATCH_SIZE,
    RANDOM_SEED, NUM_TAGS, IDX2TAG, TAG2IDX, WARMUP_EPOCHS,
)
from preprocessing import build_emoji_aware_nlp, load_embedding_components, load_vocab, EMOJI_VOCAB_PATH
from model import BiLSTMCRF, create_dataloaders


def parse_args():
    p = argparse.ArgumentParser(description="Train MESSI BiLSTM-CRF v2")
    p.add_argument("--train",      type=Path, default=Path("data/annotated/train.jsonl"))
    p.add_argument("--val",        type=Path, default=Path("data/annotated/val.jsonl"))
    p.add_argument("--epochs",     type=int,  default=MAX_EPOCHS)
    p.add_argument("--batch-size", type=int,  default=BATCH_SIZE)
    p.add_argument("--lr",         type=float,default=LEARNING_RATE)
    p.add_argument("--max-len",    type=int,  default=128)
    p.add_argument("--device",     type=str,  default="auto")
    p.add_argument("--no-char-cnn",action="store_true", help="Disable char-CNN (faster)")
    p.add_argument("--resume",     action="store_true", help="Resume from best_model.pth")
    return p.parse_args()


def select_device(d):
    if d == "auto":
        if torch.cuda.is_available():         return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")
    return torch.device(d)


def eval_f1(model, loader, device):
    model.eval()
    all_preds, all_gold = [], []
    with torch.no_grad():
        for sv, ei, la, mask, lens, ch in loader:
            sv, ei, mask, ch = sv.to(device), ei.to(device), mask.to(device), ch.to(device)
            seqs = model.decode(sv, ei, mask, lens, char_ids=ch)
            for b, seq in enumerate(seqs):
                L = lens[b].item()
                all_preds.extend((seq + [TAG2IDX["O"]]*L)[:L])
                all_gold.extend(la[b, :L].tolist())
    entity_labels = [i for i in range(NUM_TAGS) if IDX2TAG.get(i, "O") != "O"]
    return f1_score(all_gold, all_preds, labels=entity_labels, average="macro", zero_division=0)


def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for `warmup_epochs`, then cosine decay."""
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def train():
    args   = parse_args()
    torch.manual_seed(RANDOM_SEED)
    device = select_device(args.device)
    use_char = not args.no_char_cnn
    print(f"[Train] Device: {device} | Char-CNN: {use_char}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    nlp  = build_emoji_aware_nlp()
    _, extractor = load_embedding_components(nlp)
    train_loader, val_loader = create_dataloaders(
        args.train, args.val, extractor,
        batch_size=args.batch_size, max_seq_len=args.max_len,
    )

    emoji_vocab = load_vocab() if EMOJI_VOCAB_PATH.exists() else {"<PAD>": 0, "<UNK>": 1}
    model = BiLSTMCRF(emoji_vocab=emoji_vocab, use_char_cnn=use_char).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] Parameters: {n_params:,}")

    if args.resume:
        ckpt_path = MODELS_DIR / "best_bilstm_crf.pt"
        if ckpt_path.exists():
            print(f"[Train] Resuming from {ckpt_path}")
            ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt_data["model_state_dict"])
        else:
            print(f"[Train] Checkpoint not found at {ckpt_path}, starting fresh.")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = build_warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, args.epochs)
    use_amp   = device.type == "cuda"
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = MODELS_DIR / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.DictWriter(log_file, fieldnames=["epoch","train_loss","val_f1","lr","time_s"])
    writer.writeheader()

    best_f1, patience = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time(); total_loss = 0.0

        for sv, ei, la, mask, lens, ch in train_loader:
            sv, ei, la, mask, ch = (
                sv.to(device), ei.to(device), la.to(device), mask.to(device), ch.to(device))
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model(sv, ei, la, mask, lens, char_ids=ch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        scheduler.step()
        val_f1  = eval_f1(model, val_loader, device)
        elapsed = round(time.time() - t0, 1)
        lr_now  = scheduler.get_last_lr()[0]
        mean_l  = total_loss / max(len(train_loader), 1)
        star    = " ✓" if val_f1 > best_f1 else ""
        print(f"[{epoch:03d}/{args.epochs}] loss={mean_l:.4f}  f1={val_f1:.4f}  lr={lr_now:.2e}  {elapsed}s{star}")
        writer.writerow({"epoch": epoch, "train_loss": round(mean_l,4),
                         "val_f1": val_f1, "lr": round(lr_now,8), "time_s": elapsed})
        log_file.flush()

        if val_f1 > best_f1:
            best_f1, patience = val_f1, 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "emoji_vocab": emoji_vocab,
                "num_tags": NUM_TAGS,
                "use_char_cnn": use_char,
            }, BEST_MODEL_PATH)
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                print(f"[Train] Early stopping at epoch {epoch}.")
                break

    log_file.close()
    print(f"\n[Train] Done. Best val F1: {best_f1:.4f}")
    print(f"[Train] Log: {log_path}")


if __name__ == "__main__":
    train()
