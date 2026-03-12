#!/usr/bin/env python3
"""
MESSI — Web Server
==================
Flask REST API + static file server for the MESSI demo UI.

Usage:
    python server.py              # localhost:5000
    python server.py --port 8080
    python server.py --host 0.0.0.0 --port 5000  # expose on network
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "-q"])
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ── Pipeline (loaded once on startup) ─────────────────────────────────────────
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("[MESSI] Loading pipeline…", flush=True)
        from main import load_pipeline
        _pipeline = load_pipeline()
        print("[MESSI] Pipeline ready.", flush=True)
    return _pipeline


# ── Plain-English Summary Generator ───────────────────────────────────────────

_ANGRY   = {"😠","😡","🤬","💀","🔥","😤","😾","👿"}
_SAD     = {"😢","😭","💔","😞","😔","😟","🥺"}
_ALERT   = {"⚠️","🚨","🆘","❗","🔴","🛑"}
_SARCASM = {"😏","🙄","🤡","😒","🤨"}
_POLITE  = {"🙏","👍","✅","😊","❤️","🤝","😃"}
_NEUTRAL = {"😐","🤷","🤔","💭","🫤"}

def _emoji_sentiment(text: str) -> str:
    chars = set(text)
    if   chars & _ANGRY:   return "angry"
    elif chars & _ALERT:   return "alarmed"
    elif chars & _SAD:     return "upset"
    elif chars & _SARCASM: return "sarcastic"
    elif chars & _POLITE:  return "polite"
    return "neutral"

_ISSUE_PLAIN = {
    "not delivered":  "hasn't been delivered",
    "return":         "wants to return it",
    "refund":         "is asking for a refund",
    "damaged":        "arrived damaged",
    "smashed":        "arrived badly damaged",
    "wrong item":     "received the wrong item",
    "tracking":       "can't track the order",
    "payment":        "had a payment failure",
    "cancelled":      "has been cancelled",
    "delayed":        "has been delayed",
    "lost":           "has gone missing",
    "diverted":       "has been diverted",
    "baggage":        "has reported damaged baggage",
}

def generate_summary(text: str, record: dict, urgency: str, routing: str, confs: dict) -> dict:
    sentiment = _emoji_sentiment(text)
    order_id  = record.get("ORDER_ID",  record.get("order_id",  ""))
    flight_id = record.get("FLIGHT_ID", record.get("flight_id", ""))
    issue_raw = record.get("ISSUE_TYPE",record.get("issue_type","")).lower()
    event_raw = record.get("EVENT",     record.get("event",     "")).lower()

    issue_plain = issue_raw
    for key, val in _ISSUE_PLAIN.items():
        if key in issue_raw or key in text.lower():
            issue_plain = val
            break

    # Headline
    if order_id and issue_plain:
        headline = f"Customer has an issue with order {order_id} — {issue_plain}."
    elif order_id:
        headline = f"Customer is asking about order {order_id}."
    elif flight_id:
        desc = issue_plain or event_raw or "a query"
        headline = f"Flight {flight_id}: {desc}."
    else:
        headline = "Customer sent a general support message."

    sentiment_map = {
        "angry":    "They seem very frustrated and upset 😠.",
        "alarmed":  "This looks urgent and needs quick attention ⚠️.",
        "upset":    "They sound distressed and need help 😢.",
        "sarcastic":"They appear unhappy and are using sarcasm 😏.",
        "polite":   "They are being calm and polite about it 🙏.",
        "neutral":  "Their tone is calm and matter-of-fact 😐.",
    }
    sentiment_sentence = sentiment_map.get(sentiment, "")

    urgency_l = (urgency or "medium").lower()
    if urgency_l == "high":
        action_sentence = "This is HIGH priority — a support agent has been alerted and a priority ticket is being created immediately."
    elif urgency_l == "medium":
        action_sentence = "This is MEDIUM priority — a support ticket has been created and a team member will follow up shortly."
    else:
        if "auto" in (routing or "").lower():
            action_sentence = "This is LOW priority and handled automatically — the customer will receive an automated reply right away. No human agent needed."
        else:
            action_sentence = "This is LOW priority — added to the support queue."

    avg_conf = round(sum(confs.values()) / max(len(confs), 1) * 100) if confs else 0
    if avg_conf >= 85:
        conf_note = f"The AI is {avg_conf}% confident in this analysis."
    elif avg_conf >= 65:
        conf_note = f"The AI is moderately confident ({avg_conf}%) — a human may want to verify."
    else:
        conf_note = f"The AI is only {avg_conf}% confident — please have a human review this."

    meaning = " ".join(filter(None, [headline, sentiment_sentence, action_sentence, conf_note]))
    return {"headline": headline, "meaning": meaning, "sentiment": sentiment}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "BiLSTM-CRF", "version": "2.0"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    pipeline = get_pipeline()
    from main import predict as _predict
    from preprocessing import tokenize

    t0 = time.perf_counter()
    result = _predict(text, pipeline, dry_run=True)
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    # Build token + tag list for display (using direct pipeline output)
    tokens   = result.get("tokens", [])
    bio_tags = result.get("bio_tags", ["O"] * len(tokens))
    record   = result.get("record", {})

    # Build confidence list (only numeric values)
    confs = {k: round(v, 4) for k, v in result.get("confidence", {}).items()
             if isinstance(v, float)}

    urgency = result.get("urgency", "medium")
    routing = result.get("routing", "human_review")
    summary = generate_summary(text, record, urgency, routing, confs)

    return jsonify({
        "text":              text,
        "tokens":            tokens,
        "bio_tags":          bio_tags,
        "record":            record,
        "confidence":        confs,
        "urgency":           urgency,
        "action_triggered":  result.get("action_triggered", "—"),
        "routing":           routing,
        "validation_status": result.get("validation_status", "—"),
        "latency_ms":        latency_ms,
        "summary":           summary,
    })


@app.route("/examples")
def examples():
    return jsonify([
        "😠 order #4540 not delivered again asap",
        "payment failed for order #7821 🤬 third time this week",
        "great got wrong item for order #1033 😏 thanks a lot",
        "my order #2019 arrived smashed 😢 please help",
        "hi I want to return order #5678 🙏",
        "⚠️ tracking not updating on order #3312 where is it",
        "flight UA4821 delayed 🔥 been waiting 3 hours",
        "@airline DL9902 has been cancelled 🚨 what do I do",
        "AA1234 lost my bags 😭 been 2 days please",
        "LH7890 diverted 🤷 no one is telling us anything",
        "QF5501 baggage damaged 😞 suitcase completely broken",
        "😐 idk my order #9999 tracking not updating AND flight UA123 delayed",
    ])


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="MESSI web server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    # Pre-load pipeline before first request
    get_pipeline()

    print(f"\n{'═'*52}")
    print(f"  MESSI Web Server  →  http://{args.host}:{args.port}")
    print(f"{'═'*52}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
