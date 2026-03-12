"""
MESSI — Main Inference Entry Point  (flat-import version)
Usage:
    python main.py --text "order #782 not delivered 😠" --pretty
    python main.py --input-file texts.txt --output-file results.jsonl
    python main.py --pretty      (interactive REPL)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import BEST_MODEL_PATH, IDX2TAG, MC_DROPOUT_PASSES, EMOJI_VOCAB_PATH
from preprocessing import (
    build_emoji_aware_nlp, load_embedding_components,
    load_vocab, build_vocab_from_texts, save_vocab,
    tokenize,
)
from model import BiLSTMCRF
from ilp import extract_spans_from_bio, ILPValidator
from uncertainty import mc_dropout_predict, compute_confidence, overall_entropy, DecisionEngine
from api import build_output_payload, dispatch_action

_cache: dict = {}


def _auto_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_pipeline(checkpoint_path: Path = BEST_MODEL_PATH, device=None):
    global _cache
    if _cache:
        return _cache
    # Resolve "auto" or any non-torch-device string to a real device
    if device is None or str(device) in ("auto", ""):
        device = _auto_device()
    elif not isinstance(device, torch.device):
        device = torch.device(str(device))
    print(f"[MESSI] Loading pipeline on {device} …")
    t0 = time.time()

    nlp     = build_emoji_aware_nlp()
    vocab   = load_vocab() if EMOJI_VOCAB_PATH.exists() else \
              _build_demo_vocab()
    _, extractor = load_embedding_components(nlp)

    if checkpoint_path.exists():
        ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
        vocab = ckpt.get("emoji_vocab", vocab)
        use_char = ckpt.get("use_char_cnn", False)
        model = BiLSTMCRF(emoji_vocab=vocab, use_char_cnn=use_char)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[MESSI] Checkpoint loaded (val_f1={ckpt.get('val_f1','N/A')})")
    else:
        print("[MESSI] ⚠ No checkpoint — using untrained model.")
        model = BiLSTMCRF(emoji_vocab=vocab, use_char_cnn=False)

    model.to(device).eval()
    _cache = {"nlp": nlp, "extractor": extractor, "model": model,
              "device": device, "ilp": ILPValidator(), "engine": DecisionEngine()}
    print(f"[MESSI] Ready in {time.time()-t0:.1f}s")
    return _cache


def _build_demo_vocab():
    demo = ["😠", "😤", "😡", "💀", "🔥", "😊", "👍", "🙏", "🤬", "😑"]
    vocab = build_vocab_from_texts(demo)
    save_vocab(vocab, EMOJI_VOCAB_PATH)
    return vocab


def predict(text: str, pipeline: dict, dry_run: bool = True) -> dict:
    nlp, extractor = pipeline["nlp"], pipeline["extractor"]
    model, device  = pipeline["model"], pipeline["device"]
    ilp, engine    = pipeline["ilp"],   pipeline["engine"]

    tokens = tokenize(text, nlp)
    if not tokens:
        return {"error": "empty input", "raw_input": text}

    sv, ei   = extractor.batch_extract([tokens])
    sv, ei   = sv.to(device), ei.to(device)
    lengths  = torch.tensor([len(tokens)], dtype=torch.long)
    mask     = torch.ones(1, len(tokens), dtype=torch.bool, device=device)

    # Char ids for char-CNN
    from model import tokens_to_char_ids
    char_ids = tokens_to_char_ids(tokens).unsqueeze(0).to(device)  # (1, L, W)

    best_tags, entropies = mc_dropout_predict(model, sv, ei, mask, T=MC_DROPOUT_PASSES,
                                              lengths=lengths, char_ids=char_ids)
    bio_tags = [IDX2TAG.get(i, "O") for i in best_tags]

    spans      = extract_spans_from_bio(tokens, bio_tags)
    ilp_result = ilp.solve(spans)
    confidences = compute_confidence(entropies)
    oe          = overall_entropy(entropies)

    decision = engine.decide(
        record=ilp_result["record"], confidences=confidences,
        entropies=entropies, raw_text=text,
        validation_status=ilp_result["validation_status"],
    )
    decision["confidence"]["overall_entropy"] = oe

    output = build_output_payload(decision, raw_text=text)
    output["bio_tags"] = bio_tags
    output["tokens"]   = tokens
    return dispatch_action(output, dry_run=dry_run)


def main():
    p = argparse.ArgumentParser(description="MESSI inference")
    p.add_argument("--text",         default=None)
    p.add_argument("--input-file",   type=Path, default=None)
    p.add_argument("--output-file",  type=Path, default=None)
    p.add_argument("--checkpoint",   type=Path, default=BEST_MODEL_PATH)
    p.add_argument("--dispatch",     action="store_true")
    p.add_argument("--device",       default="auto")
    p.add_argument("--pretty",       action="store_true")
    args = p.parse_args()

    device   = None if args.device == "auto" else torch.device(args.device)
    pipeline = load_pipeline(args.checkpoint, device)
    dry_run  = not args.dispatch
    indent   = 2 if args.pretty else None

    if args.text:
        print(json.dumps(predict(args.text, pipeline, dry_run), indent=indent, ensure_ascii=False))
    elif args.input_file:
        with open(args.input_file) as f:
            texts = [l.strip() for l in f if l.strip()]
        results = [json.dumps(predict(t, pipeline, dry_run), ensure_ascii=False) for t in texts]
        if args.output_file:
            with open(args.output_file, "w") as out:
                out.write("\n".join(results))
            print(f"[MESSI] {len(results)} results → {args.output_file}")
        else:
            print("\n".join(results))
    else:
        print("MESSI Interactive (Ctrl-C to exit)\n" + "="*50)
        try:
            while True:
                t = input(">>> ").strip()
                if t:
                    print(json.dumps(predict(t, pipeline, dry_run), indent=2, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\n[MESSI] Bye!")


if __name__ == "__main__":
    main()
