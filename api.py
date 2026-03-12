"""
MESSI — Layer 6: API Integration & Payload Builder
====================================================
Merges: payload_builder.py + all 6 integration stubs

Exported:
  build_output_payload(decision, raw_text)  → dict
  dispatch_action(payload, dry_run)         → dict
  (stub functions for Zendesk, Salesforce, Slack, Twilio, Firebase, Postgres)
"""

import time
from typing import Dict, Optional

import requests

from config import (
    ZENDESK_URL, ZENDESK_EMAIL, ZENDESK_TOKEN,
    SALESFORCE_URL, SALESFORCE_TOKEN,
    SLACK_WEBHOOK_URL,
    TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM,
    FIREBASE_URL,
    POSTGRES_DSN,
    API_TIMEOUT_SECS, API_MAX_RETRIES,
)


# ═══════════════════════════════════════════════════════════════
#  Payload Builder
# ═══════════════════════════════════════════════════════════════

_ACTION_LABELS = {
    "zendesk_ticket":         "zendesk_ticket_created",
    "salesforce_case":        "salesforce_case_created",
    "notify_passenger_slack": "slack_notification_sent",
    "twilio_sms":             "sms_sent",
    "firebase_log":           "firebase_event_logged",
    "log_to_postgres":        "postgres_record_logged",
    "human_review_queue":     "routed_to_human_review",
    "unknown":                "no_action_taken",
}


def build_output_payload(decision: Dict, raw_text: str = "") -> Dict:
    """Convert DecisionEngine output → Blueprint §5 JSON schema."""
    record    = decision.get("record", {})
    confidence = decision.get("confidence", {})
    action    = decision.get("action_triggered", "unknown")

    norm_conf = {}
    for field, score in confidence.items():
        if field == "overall_entropy":
            norm_conf["overall_entropy"] = score
        else:
            norm_conf[field.lower()] = score

    payload = {
        "raw_input":         raw_text,
        "validation_status": decision.get("validation_status", "Unknown"),
        "urgency":           decision.get("urgency", "unknown"),
        "action_triggered":  _ACTION_LABELS.get(action, action),
        "routing":           decision.get("routing", "automated"),
        "confidence":        norm_conf,
    }
    for field, text in record.items():
        payload[field.lower()] = text
    if "order_id" in payload:
        payload["entity_id"] = payload.pop("order_id")
    if "flight_id" in payload:
        payload["entity_id"] = payload.pop("flight_id")
    return payload


def dispatch_action(payload: Dict, dry_run: bool = False) -> Dict:
    """Route the payload to the correct integration stub."""
    if dry_run:
        payload["api_response"] = {
            "status": "dry_run",
            "message": f"Would have triggered: {payload.get('action_triggered')}",
        }
        return payload

    label = payload.get("action_triggered", "")
    _reverse = {v: k for k, v in _ACTION_LABELS.items()}
    key  = _reverse.get(label, label)

    _dispatch = {
        "zendesk_ticket":         _zendesk_ticket,
        "salesforce_case":        _salesforce_case,
        "notify_passenger_slack": _slack_message,
        "twilio_sms":             _twilio_sms,
        "firebase_log":           _firebase_log,
        "log_to_postgres":        _postgres_insert,
        "human_review_queue":     _postgres_review_queue,
    }
    fn = _dispatch.get(key)
    if fn:
        try:
            payload["api_response"] = fn(payload)
        except Exception as exc:
            payload["api_response"] = {"status": "error", "message": str(exc)}
    else:
        payload["api_response"] = {"status": "no_integration", "message": f"No handler: {key}"}
    return payload


# ═══════════════════════════════════════════════════════════════
#  Integration Stubs
# ═══════════════════════════════════════════════════════════════

def _retry(fn, retries=API_MAX_RETRIES):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (i + 1))


def _zendesk_ticket(payload: Dict) -> Dict:
    body = {
        "ticket": {
            "subject":  f"[MESSI] {payload.get('action_triggered', 'Issue')} — {payload.get('entity_id', '?')}",
            "comment":  {"body": payload.get("raw_input", "")},
            "priority": payload.get("urgency", "normal"),
        }
    }
    def _call():
        r = requests.post(
            f"{ZENDESK_URL}/api/v2/tickets.json",
            json=body,
            auth=(f"{ZENDESK_EMAIL}/token", ZENDESK_TOKEN),
            timeout=API_TIMEOUT_SECS,
        )
        r.raise_for_status()
        return {"status": "created", "ticket_id": r.json().get("ticket", {}).get("id")}
    return _retry(_call)


def _salesforce_case(payload: Dict) -> Dict:
    body = {
        "Subject":      f"[MESSI] {payload.get('issue_type', 'Issue')}",
        "Description":  payload.get("raw_input", ""),
        "Priority":     "High" if payload.get("urgency") == "high" else "Medium",
        "Status":       "New",
    }
    def _call():
        r = requests.post(
            f"{SALESFORCE_URL}/services/data/v57.0/sobjects/Case/",
            json=body,
            headers={"Authorization": f"Bearer {SALESFORCE_TOKEN}"},
            timeout=API_TIMEOUT_SECS,
        )
        r.raise_for_status()
        return {"status": "created", "case_id": r.json().get("id")}
    return _retry(_call)


def _slack_message(payload: Dict) -> Dict:
    msg = (f"*[MESSI Alert]* `{payload.get('action_triggered')}` | "
           f"urgency=`{payload.get('urgency')}` | "
           f"input=_{payload.get('raw_input', '')}_")
    def _call():
        r = requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=API_TIMEOUT_SECS)
        r.raise_for_status()
        return {"status": "sent"}
    return _retry(_call)


def _twilio_sms(payload: Dict) -> Dict:
    body = (f"[MESSI] {payload.get('urgency', '').upper()} urgency alert: "
            f"{payload.get('action_triggered')} — {payload.get('raw_input', '')[:80]}")
    def _call():
        r = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
            data={"Body": body, "From": TWILIO_FROM, "To": "+10000000001"},
            auth=(TWILIO_SID, TWILIO_TOKEN),
            timeout=API_TIMEOUT_SECS,
        )
        r.raise_for_status()
        return {"status": "sent", "sid": r.json().get("sid")}
    return _retry(_call)


def _firebase_log(payload: Dict) -> Dict:
    import time
    entry = {**payload, "logged_at": time.time()}
    def _call():
        r = requests.post(
            f"{FIREBASE_URL}/messi_logs.json",
            json=entry, timeout=API_TIMEOUT_SECS,
        )
        r.raise_for_status()
        return {"status": "logged", "firebase_key": r.json().get("name")}
    return _retry(_call)


def _postgres_insert(payload: Dict) -> Dict:
    try:
        import psycopg2, json as _json
        conn = psycopg2.connect(POSTGRES_DSN)
        cur  = conn.cursor()
        cur.execute(
            """INSERT INTO messi_records
               (entity_id, issue_type, urgency, confidence, validation_status, action_triggered, raw_input)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (payload.get("entity_id"), payload.get("issue_type"),
             payload.get("urgency"), _json.dumps(payload.get("confidence", {})),
             payload.get("validation_status"), payload.get("action_triggered"),
             payload.get("raw_input")),
        )
        conn.commit(); conn.close()
        return {"status": "inserted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _postgres_review_queue(payload: Dict) -> Dict:
    try:
        import psycopg2, json as _json
        conn = psycopg2.connect(POSTGRES_DSN)
        cur  = conn.cursor()
        cur.execute(
            "INSERT INTO human_review_queue (payload, reason) VALUES (%s, %s)",
            (_json.dumps(payload), "MC Dropout entropy threshold exceeded"),
        )
        conn.commit(); conn.close()
        return {"status": "queued"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
