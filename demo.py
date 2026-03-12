#!/usr/bin/env python3
"""
MESSI — Live Inference Demo
============================
Showcases the full 6-layer pipeline on hand-crafted messages from both domains.
Prints a rich, coloured report for every message: tokens → NER → ILP → confidence
→ urgency → action decision.

Usage:
    python demo.py                    # run all 12 showcase examples
    python demo.py --text "my msg"    # single custom message
    python demo.py --no-color         # plain text (for logs / CI)
    python demo.py --live             # actually call APIs (default: dry-run)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Colour helpers ─────────────────────────────────────────────────────────────
# NOTE: lazy functions so --no-color set in main() takes effect everywhere.
_USE_COLOR = True

def _R():  return "\033[0m"  if _USE_COLOR else ""
def _B():  return "\033[1m"  if _USE_COLOR else ""
def _D():  return "\033[2m"  if _USE_COLOR else ""
def _RE(): return "\033[91m" if _USE_COLOR else ""  # red
def _GR(): return "\033[92m" if _USE_COLOR else ""  # green
def _YE(): return "\033[93m" if _USE_COLOR else ""  # yellow
def _BL(): return "\033[94m" if _USE_COLOR else ""  # blue
def _MA(): return "\033[95m" if _USE_COLOR else ""  # magenta
def _CY(): return "\033[96m" if _USE_COLOR else ""  # cyan
def _WH(): return "\033[97m" if _USE_COLOR else ""  # white


def banner(text, width=66, color_fn=_BL):
    bar = "═" * width
    c = color_fn()
    print(f"\n{_B()}{c}{bar}{_R()}")
    print(f"{_B()}{c}  {text}{_R()}")
    print(f"{_B()}{c}{bar}{_R()}")


def section(label, color_fn=_CY):
    print(f"\n  {_B()}{color_fn()}▶ {label}{_R()}")


def kv(key, val):
    print(f"    {_D()}{key:<22}{_R()} {val}{_R()}")


def tag_colored(tag):
    if tag == "O":            return f"{_D()}O{_R()}"
    if tag.startswith("B-"): return f"{_GR()}{tag}{_R()}"
    if tag.startswith("I-"): return f"{_CY()}{tag}{_R()}"
    return tag


def urgency_color(u):
    return {
        "high":   f"{_B()}{_RE()}HIGH   🔴{_R()}",
        "medium": f"{_B()}{_YE()}MEDIUM 🟡{_R()}",
        "low":    f"{_B()}{_GR()}LOW    🟢{_R()}",
    }.get(u, u)


def routing_color(r):
    return {
        "automated":    f"{_GR()}✅ automated{_R()}",
        "human_review": f"{_YE()}👤 human review{_R()}",
    }.get(r, r)


# ── Showcase messages ──────────────────────────────────────────────────────────
SHOWCASE = [
    # E-commerce — angry
    "😠 order #4540 not delivered again asap",
    # E-commerce — payment
    "payment failed for order #7821 🤬 third time this week",
    # E-commerce — wrong item (sarcasm)
    "great got wrong item for order #1033 😏 thanks a lot",
    # E-commerce — damaged (sad)
    "my order #2019 arrived smashed 😢 please help",
    # E-commerce — return (polite)
    "hi I want to return order #5678 🙏",
    # E-commerce — tracking alert
    "⚠️ tracking not updating on order #3312 where is it",
    # Aviation — delayed (angry)
    "flight UA4821 delayed 🔥 been waiting 3 hours",
    # Aviation — cancelled (alert)
    "@airline DL9902 has been cancelled 🚨 what do I do",
    # Aviation — missing bags (sad)
    "AA1234 lost my bags 😭 been 2 days please",
    # Aviation — diverted (neutral)
    "LH7890 diverted 🤷 no one is telling us anything",
    # Aviation — damaged (sad)
    "QF5501 baggage damaged 😞 suitcase completely broken",
    # Mixed / ambiguous
    "😐 idk my order #9999 tracking not updating AND flight UA123 delayed",
]

# ── Pipeline loader (cached) ───────────────────────────────────────────────────
_pipeline = None


def get_pipeline(device="auto"):
    global _pipeline
    if _pipeline is None:
        print(f"\n{_B()}{_MA()}Loading MESSI pipeline…{_R()}", end=" ", flush=True)
        from main import load_pipeline
        _pipeline = load_pipeline(device=device)
        print(f"{_GR()}ready{_R()}")
    return _pipeline


# ── Single message runner ──────────────────────────────────────────────────────

def run_message(text: str, pipeline: dict, idx: int, dry_run: bool = True):
    from main import predict
    from preprocessing import tokenize

    print(f"\n{_B()}{_BL()}{'─'*66}{_R()}")
    print(f"  {_B()}[{idx}] {_WH()}{text}{_R()}")

    t0 = time.perf_counter()
    result = predict(text, pipeline, dry_run=dry_run)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if "error" in result:
        print(f"  {_RE()}⚠  Error: {result['error']}{_R()}")
        return

    # ── Tokens + actual BIO Predictions (from neural network) ─────────────────
    section("Neural Tagging (BiLSTM-CRF)")
    tokens   = result.get("tokens", [])
    bio_tags = result.get("bio_tags", ["O"] * len(tokens))
    
    token_str = "  ".join(
        f"{tok}/{tag_colored(tag)}" for tok, tag in zip(tokens, bio_tags)
    )
    print(f"    {token_str}")

    # ── Extracted record ───────────────────────────────────────────────────────
    section("Extracted Entities  (post-ILP)")
    # Flattened payload keys (from build_output_payload)
    fields = ["entity_id", "issue_type", "event", "time"]
    found = False
    for f in fields:
        val = result.get(f)
        if val:
            kv(f.replace("_"," ").title(), f"{_GR()}{val}{_R()}")
            found = True
    if not found:
        print(f"    {_D()}No entities extracted{_R()}")

    # ── Confidence  ────────────────────────────────────────────────────────────
    section("MC Dropout Confidence")
    confs = {k: v for k, v in result.get("confidence", {}).items()
             if isinstance(v, float)}
    for field, conf in confs.items():
        bar_len = int(max(0.0, min(1.0, conf)) * 20)
        bar = f"{_GR()}{'█' * bar_len}{_D()}{'░' * (20 - bar_len)}{_R()}"
        kv(field, f"{bar}  {conf:.2%}")
    oe = result.get("confidence", {}).get("overall_entropy", "—")
    kv("overall entropy H", str(oe))

    # ── Decision  ──────────────────────────────────────────────────────────────
    section("Decision Engine Output")
    kv("Urgency",          urgency_color(result.get("urgency", "?")))
    kv("Action triggered", f"{_MA()}{result.get('action_triggered', '—')}{_R()}")
    kv("Routing",          routing_color(result.get("routing", "?")))
    kv("ILP status",       result.get("validation_status", "—"))
    kv("Latency",          f"{elapsed_ms:.1f} ms")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _USE_COLOR

    p = argparse.ArgumentParser(description="MESSI live demo")
    p.add_argument("--text",     default=None, help="Single message to analyse")
    p.add_argument("--device",   default="auto")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colours")
    p.add_argument("--live",     action="store_true",
                   help="Call real APIs (default: dry-run)")
    args = p.parse_args()

    # Must set BEFORE any color function is called
    if args.no_color:
        _USE_COLOR = False

    banner("MESSI — Neuro-Symbolic NER & Decision Demo", color_fn=_MA)
    print(f"  {_D()}6-layer pipeline: Tokenise → BiLSTM-CRF → ILP → MC Dropout → Engine → API{_R()}")

    pipeline = get_pipeline(args.device)
    dry_run  = not args.live
    messages = [args.text] if args.text else SHOWCASE

    for i, msg in enumerate(messages, 1):
        run_message(msg, pipeline, idx=i, dry_run=dry_run)

    banner(f"Done  —  {len(messages)} messages processed", color_fn=_GR)
    if dry_run:
        print(f"  {_D()}(dry-run mode — APIs not called. Use --live to enable real dispatch){_R()}\n")


if __name__ == "__main__":
    main()
