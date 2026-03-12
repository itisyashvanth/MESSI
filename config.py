"""
MESSI — Central Configuration
All hyperparameters, file paths, and thresholds are defined here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
ANNOTATED_DIR = DATA_DIR / "annotated"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"

BEST_MODEL_PATH = MODELS_DIR / "best_bilstm_crf.pt"
EMOJI_VOCAB_PATH = DATA_DIR / "emoji_vocab.json"

# ─────────────────────────────────────────────
#  NLP
# ─────────────────────────────────────────────
SPACY_MODEL = "en_core_web_md"
SPACY_VECTOR_DIM = 300          # en_core_web_md output dimension

# ─────────────────────────────────────────────
#  Embedding
# ─────────────────────────────────────────────
EMOJI_EMBEDDING_DIM = 50        # Trainable emoji embedding size
UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"

# ─────────────────────────────────────────────
#  BiLSTM-CRF Architecture
# ─────────────────────────────────────────────
LSTM_HIDDEN_DIM = 320          # ↑ from 256 → more capacity
LSTM_LAYERS = 3               # ↑ from 2 → deeper encoder
LSTM_DROPOUT = 0.25           # ↓ from 0.35 (relaxed for smaller real dataset)
EMBEDDING_DIM = SPACY_VECTOR_DIM + EMOJI_EMBEDDING_DIM  # 350
EMBEDDING_DROPOUT = 0.20      # ↓ from 0.25

# Character-level CNN (optional — set False to disable for speed)
USE_CHAR_CNN = True
CHAR_VOCAB_SIZE = 128          # ASCII printable set
CHAR_EMB_DIM = 32
CHAR_CNN_FILTERS = 64
CHAR_CNN_KERNEL = 3

# ─────────────────────────────────────────────
#  Training
# ─────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 5e-4           # ↓ from 1e-3 (cosine schedule)
WARMUP_EPOCHS = 2              # linear warmup before cosine decay
MAX_EPOCHS = 70                # ↑ from 60
GRAD_CLIP_MAX_NORM = 5.0
EARLY_STOPPING_PATIENCE = 15   # ↑ from 10
LABEL_SMOOTHING = 0.08         # CRF label smoothing coefficient
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
RANDOM_SEED = 42

# ─────────────────────────────────────────────
#  BIO Tag Schema
# ─────────────────────────────────────────────
# Aviation domain tags
AVIATION_TAGS = [
    "O",
    "B-FLIGHT_ID", "I-FLIGHT_ID",
    "B-EVENT",     "I-EVENT",
    "B-TIME",      "I-TIME",
]

# E-Commerce / Support domain tags
ECOMMERCE_TAGS = [
    "O",
    "B-ORDER_ID",    "I-ORDER_ID",
    "B-ISSUE_TYPE",  "I-ISSUE_TYPE",
]

# Combined tag set (used for joint training)
ALL_TAGS = sorted(set(AVIATION_TAGS + ECOMMERCE_TAGS))
TAG2IDX = {tag: idx for idx, tag in enumerate(ALL_TAGS)}
IDX2TAG = {idx: tag for tag, idx in TAG2IDX.items()}
NUM_TAGS = len(ALL_TAGS)

PAD_TAG_IDX = TAG2IDX["O"]     # CRF uses O as the pad label

# ─────────────────────────────────────────────
#  ILP Validation Rules
# ─────────────────────────────────────────────
FIELD_REGEX_RULES = {
    # e.g. UA123, DL9969, QF4218 — 2 lowercase or uppercase letters + 2-4 digits
    "FLIGHT_ID": r"^[A-Z]{2}\d{2,4}$",
    # ORDER_ID can have a leading hash or just digits
    "ORDER_ID":  r"^#?\d+$",
}

# Surface phrases that can appear in the bio-tagged span for each field.
# These are the actual surface expressions used by data_utils.py.
FIELD_ALLOWLISTS = {
    "EVENT": [
        # delayed
        "delayed", "running late", "late departure", "pushed back",
        # cancelled
        "cancelled", "canceled", "has been cancelled",
        # diverted
        "diverted", "rerouted", "went to wrong airport",
        # missing baggage
        "baggage missing", "lost my bags", "luggage lost",
        "bags not arrived", "baggage missing",
        # damaged
        "bag is broken", "baggage damaged", "luggage destroyed",
    ],
    "ISSUE_TYPE": [
        "not delivered", "never arrived", "missing", "didn't receive",
        "where is my order",
        "payment failed", "charge declined", "couldn't pay",
        "payment error", "payment not processed",
        "wrong item", "incorrect product", "not what I ordered", "got wrong thing",
        "damaged", "broken", "arrived smashed", "item is damaged",
        "want to return", "return this", "send it back", "need a return",
        "no tracking", "tracking not updating", "can't track my order",
        "tracking info missing",
    ],
}

FIELD_CARDINALITY = {
    # (min, max) occurrences per input
    "FLIGHT_ID":  (0, 1),
    "ORDER_ID":   (0, 1),
    "EVENT":      (0, 2),
    "ISSUE_TYPE": (0, 3),
}

# ─────────────────────────────────────────────
#  Uncertainty Estimation
# ─────────────────────────────────────────────
MC_DROPOUT_PASSES = 10          # Number of stochastic forward passes
ENTROPY_THRESHOLD = 0.5         # H > threshold → route to human review

# ─────────────────────────────────────────────
#  Decision Engine Thresholds
# ─────────────────────────────────────────────
HIGH_URGENCY_KEYWORDS = [
    # Text keywords
    "urgent", "immediately", "asap", "now", "emergency",
    # ANGER emojis (most frequent, highest urgency)
    "😠", "😡", "🤬", "💀", "🔥",
    # ALERT emojis (high urgency signals)
    "⚠️", "🚨", "🔔", "⏰", "📢", "🆘",
]

# Medium urgency — sadness, sarcasm, mild frustration
MEDIUM_URGENCY_EMOJIS = [
    # MILD
    "😑", "🙄", "😒", "😤",
    # SADNESS
    "😢", "😭", "💔", "😔", "😞", "😿", "🥺", "😩", "😫",
    # SARCASM
    "😏", "🤡", "🥴", "💅",
]

# Low urgency / positive — polite requests or resolved issues
LOW_URGENCY_EMOJIS = [
    "👍", "😊", "🙏", "✅", "🎉", "😄", "🤗", "💯", "🙌",
    "🤷", "😐", "🤔", "🫤", "😶", "🧐",
]

CONFIDENCE_LOW_THRESHOLD = 0.70  # Below this → low confidence

# Issue → Action mapping
ISSUE_ACTION_MAP = {
    "not_delivered":   "zendesk_ticket",
    "payment_failed":  "salesforce_case",
    "wrong_item":      "zendesk_ticket",
    "damaged_item":    "zendesk_ticket",
    "return_request":  "salesforce_case",
    "account_issue":   "salesforce_case",
    "refund_pending":  "zendesk_ticket",
    "tracking_missing":"zendesk_ticket",
}

EVENT_ACTION_MAP = {
    "delayed":          "notify_passenger_slack",
    "cancelled":        "notify_passenger_slack",
    "missing_baggage":  "zendesk_ticket",
    "damaged":          "zendesk_ticket",
}

# ─────────────────────────────────────────────
#  API Integration
# ─────────────────────────────────────────────
# Override via environment variables in production
ZENDESK_URL       = os.getenv("ZENDESK_URL",       "https://your-domain.zendesk.com")
ZENDESK_EMAIL     = os.getenv("ZENDESK_EMAIL",     "admin@example.com")
ZENDESK_TOKEN     = os.getenv("ZENDESK_TOKEN",     "ZENDESK_API_TOKEN")

SALESFORCE_URL    = os.getenv("SALESFORCE_URL",    "https://your-domain.salesforce.com")
SALESFORCE_TOKEN  = os.getenv("SALESFORCE_TOKEN",  "SALESFORCE_API_TOKEN")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/XXX/YYY/ZZZ")
TWILIO_SID        = os.getenv("TWILIO_SID",        "TWILIO_ACCOUNT_SID")
TWILIO_TOKEN      = os.getenv("TWILIO_TOKEN",      "TWILIO_AUTH_TOKEN")
TWILIO_FROM       = os.getenv("TWILIO_FROM",       "+10000000000")

FIREBASE_URL      = os.getenv("FIREBASE_URL",      "https://your-project.firebaseio.com")
POSTGRES_DSN      = os.getenv("POSTGRES_DSN",      "postgresql://user:pass@localhost:5432/messi")

API_TIMEOUT_SECS  = 10
API_MAX_RETRIES   = 3
