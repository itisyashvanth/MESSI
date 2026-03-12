"""
MESSI — Python Package Setup
=============================
Install with:  pip install -e .
Run server:    python -m messi.server
Run demo:      python -m messi.demo --text "your message"
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name             = "messi-ner",
    version          = "2.0.0",
    description      = "MESSI — Neuro-Symbolic NER & Decision Automation System",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author           = "MESSI Team",
    python_requires  = ">=3.10",

    # ── Source ───────────────────────────────────────────────────────
    packages         = find_packages(exclude=["tests*", "*.tests"]),
    py_modules       = [
        # Root-level flat modules (not in a sub-package)
        "config",
        "preprocessing",
        "model",
        "train",
        "evaluate",
        "benchmark",
        "demo",
        "main",
        "ilp",
        "uncertainty",
        "api",
        "data_utils",
        "download_datasets",
        "run_all",
        "server",
        "baselines",
        "progress_check",
    ],

    # ── Static data files bundled with the package ───────────────────
    package_data     = {
        "": [
            "static/*.html",
            "static/*.js",
            "static/*.css",
            "data/emoji_vocab.json",
            "data/annotated/*.jsonl",
            "models/best_bilstm_crf.pt",
            "requirements.txt",
            "pytest.ini",
        ]
    },
    include_package_data = True,

    # ── Dependencies ─────────────────────────────────────────────────
    install_requires = [
        "spacy>=3.7.4",
        "emoji>=2.10.1",
        "torch>=2.2.1",
        "ortools>=9.9.3963",
        "numpy>=1.26.4",
        "pandas>=2.2.1",
        "scikit-learn>=1.4.1",
        "tqdm>=4.66.2",
        "pyyaml>=6.0.1",
        "jsonlines>=4.0.0",
        "pydantic>=2.0",
        "requests>=2.31.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "datasets>=2.18.0",
    ],
    extras_require   = {
        "baselines": ["transformers>=4.39.3", "accelerate>=0.28.0"],
        "dev":       ["pytest>=8.1.1", "pytest-cov>=5.0.0"],
        "notebook":  ["jupyter>=1.0.0", "ipykernel>=6.29.4"],
    },

    # ── Entry points (CLI commands) ───────────────────────────────────
    entry_points     = {
        "console_scripts": [
            "messi         = main:main",
            "messi-server  = server:main",
            "messi-train   = train:train",
            "messi-eval    = evaluate:main",
            "messi-bench   = benchmark:main",
            "messi-demo    = demo:main",
            "messi-run-all = run_all:main",
        ]
    },

    classifiers      = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
