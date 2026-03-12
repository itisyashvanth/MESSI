#!/usr/bin/env python3
"""
MESSI — run_all.py
==================
One-command end-to-end pipeline orchestrator.

Stages:
  1. Data generation  (data_utils.py)
  2. Training         (train.py)
  3. Evaluation       (evaluate.py)
  4. Benchmark        (benchmark.py)
  5. Demo             (demo.py)
  6. Summary report

Usage:
    python run_all.py                          # full run, all stages
    python run_all.py --skip-train             # use existing checkpoint
    python run_all.py --quick                  # fast settings (5 epochs, 500 samples)
    python run_all.py --stages data,train,eval # run specific stages only
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Colours ────────────────────────────────────────────────────────────────────
G  = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
B  = "\033[94m"; M = "\033[95m"; BOLD = "\033[1m"; RST = "\033[0m"; DIM = "\033[2m"

def hdr(title, emoji="", color=B):
    bar = "═" * 64
    print(f"\n{BOLD}{color}{bar}{RST}")
    print(f"{BOLD}{color}  {emoji}  {title}{RST}")
    print(f"{BOLD}{color}{bar}{RST}")

def step(msg):  print(f"\n  {BOLD}{Y}▶  {msg}{RST}")
def ok(msg):    print(f"  {G}✓  {msg}{RST}")
def fail(msg):  print(f"  {R}✗  {msg}{RST}")
def info(msg):  print(f"  {DIM}   {msg}{RST}")

# ── Stage runner ───────────────────────────────────────────────────────────────

def run_stage(name: str, cmd: list, cwd=None) -> bool:
    step(f"Stage: {name}")
    info(" ".join(str(c) for c in cmd))
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable] + [str(c) for c in cmd],
            cwd=cwd or Path(__file__).parent,
            check=True,
        )
        elapsed = time.time() - t0
        ok(f"{name} completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        fail(f"{name} failed (exit {e.returncode})")
        return False
    except Exception as e:
        fail(f"{name} error: {e}")
        return False


# ── Stage definitions ──────────────────────────────────────────────────────────

def stage_download(args):
    """Download + convert real datasets from HuggingFace."""
    return run_stage(
        "Dataset Download",
        ["download_datasets.py"],
    )


def stage_data(args):
    """Generate synthetic training data."""
    n_ec = 500 if args.quick else 3000
    n_av = 300 if args.quick else 2000
    return run_stage(
        "Data Generation",
        ["data_utils.py", f"--n-ec={n_ec}", f"--n-av={n_av}"],
    )


def stage_train(args):
    """Train the BiLSTM-CRF model."""
    epochs = 5 if args.quick else 60
    return run_stage(
        "Model Training",
        ["train.py", f"--epochs={epochs}",
         "--batch-size=32", "--device=auto"],
    )


def stage_eval(args):
    """Evaluate on test split."""
    return run_stage(
        "Evaluation",
        ["evaluate.py", "--test=data/annotated/test.jsonl", "--show-errors"],
    )


def stage_benchmark(args):
    """Run efficiency + accuracy benchmark."""
    epochs  = 5  if args.quick else 20
    samples = 300 if args.quick else 1000
    return run_stage(
        "Benchmark",
        ["benchmark.py", f"--epochs={epochs}", f"--samples={samples}"],
    )


def stage_demo(args):
    """Run live inference demo (dry-run)."""
    return run_stage("Live Demo", ["demo.py", "--no-color"])


ALL_STAGES = {
    "download":  stage_download,
    "data":      stage_data,
    "train":     stage_train,
    "eval":      stage_eval,
    "benchmark": stage_benchmark,
    "demo":      stage_demo,
}

# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(results: dict, total_time: float):
    hdr("Run Summary", "📊", color=M)
    for name, ok_ in results.items():
        icon  = f"{G}✓{RST}" if ok_ else f"{R}✗{RST}"
        label = f"{G}PASS{RST}" if ok_ else f"{R}FAIL{RST}"
        print(f"  {icon}  {name:<18} {label}")
    passed = sum(results.values())
    total  = len(results)
    color  = G if passed == total else (Y if passed > 0 else R)
    print(f"\n  {BOLD}{color}{passed}/{total} stages passed  —  {total_time:.1f}s total{RST}\n")

    if passed == total:
        print(f"  {BOLD}{G}🎉 MESSI pipeline complete!{RST}")
        print(f"\n  Next steps:")
        print(f"  {DIM}  python demo.py --text \"my message here\"{RST}")
        print(f"  {DIM}  python demo.py  (12 showcase messages){RST}\n")
    else:
        print(f"  {R}Some stages failed. Check output above.{RST}\n")


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MESSI end-to-end pipeline runner")
    p.add_argument("--quick",       action="store_true",
                   help="Fast run: 5 epochs, 800 samples (for testing)")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip dataset download (use existing data)")
    p.add_argument("--skip-train",  action="store_true",
                   help="Skip training (use existing checkpoint)")
    p.add_argument("--skip-data",   action="store_true",
                   help="Skip data generation (use existing annotated/ files)")
    p.add_argument("--stages",      default=None,
                   help="Comma-separated list of stages: data,train,eval,benchmark,demo")
    args = p.parse_args()

    # Determine which stages to run
    if args.stages:
        selected = [s.strip() for s in args.stages.split(",")]
    else:
        selected = list(ALL_STAGES.keys())
        if args.skip_download: selected.remove("download")
        if args.skip_data:  selected.remove("data")
        if args.skip_train: selected.remove("train")

    hdr("MESSI — End-to-End Pipeline Orchestrator", "🚀", color=M)
    print(f"  {DIM}Stages: {', '.join(selected)}{RST}")
    print(f"  {DIM}Mode:   {'quick' if args.quick else 'full'}{RST}")

    t0      = time.time()
    results = {}

    for stage_name in selected:
        if stage_name not in ALL_STAGES:
            fail(f"Unknown stage '{stage_name}'. Valid: {', '.join(ALL_STAGES)}")
            results[stage_name] = False
            continue
        success = ALL_STAGES[stage_name](args)
        results[stage_name] = success
        if not success and stage_name in ("data", "train"):
            fail(f"Critical stage '{stage_name}' failed — aborting remaining stages.")
            for remaining in selected[list(selected).index(stage_name)+1:]:
                results[remaining] = False
            break

    print_summary(results, time.time() - t0)


if __name__ == "__main__":
    main()
