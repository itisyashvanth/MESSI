"""
MESSI — Layers 4+5: Uncertainty Estimation & Decision Engine
=============================================================
Merges: mc_dropout.py + entropy.py + engine.py

Exported:
  predictive_entropy(counts, T)       → float
  mc_dropout_predict(model, ...)      → (best_tags, entropies)
  compute_confidence(entropies)       → Dict[str, float]
  overall_entropy(entropies)          → float
  should_route_to_human(entropies)    → bool
  DecisionEngine.decide(...)          → decision dict
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import (
    MC_DROPOUT_PASSES, ENTROPY_THRESHOLD, NUM_TAGS, IDX2TAG, TAG2IDX,
    HIGH_URGENCY_KEYWORDS, MEDIUM_URGENCY_EMOJIS, LOW_URGENCY_EMOJIS,
    CONFIDENCE_LOW_THRESHOLD, ISSUE_ACTION_MAP, EVENT_ACTION_MAP,
)


# ═══════════════════════════════════════════════════════════════
#  Entropy Utilities
# ═══════════════════════════════════════════════════════════════

def predictive_entropy(counts: Dict[str, int], T: int) -> float:
    """H(y_f) = −Σ p̂_c log(p̂_c)  where p̂_c = count_c / T"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    H = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            H -= p * math.log(p + 1e-12)
    return H


# ═══════════════════════════════════════════════════════════════
#  MC Dropout
# ═══════════════════════════════════════════════════════════════

def _enable_mc_dropout(model: nn.Module) -> None:
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()


def mc_dropout_predict(
    model: nn.Module,
    spacy_vectors: torch.Tensor,
    emoji_ids: torch.Tensor,
    mask: torch.Tensor,
    T: int = MC_DROPOUT_PASSES,
    lengths: Optional[torch.Tensor] = None,
    char_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """
    Run T stochastic forward passes → majority-vote tags + per-field entropy.

    Returns:
        best_tags : List[int]         — most frequent tag per position
        entropies : Dict[str, float]  — predictive entropy per entity field
    """
    _enable_mc_dropout(model)
    all_tag_seqs: List[List[int]] = []

    with torch.no_grad():
        for _ in range(T):
            tags = model.decode(spacy_vectors, emoji_ids, mask, lengths, char_ids=char_ids)
            all_tag_seqs.append(tags[0])

    seq_len = len(all_tag_seqs[0])

    # Majority vote
    best_tags: List[int] = []
    for pos in range(seq_len):
        votes = Counter(seq[pos] if pos < len(seq) else 0 for seq in all_tag_seqs)
        best_tags.append(votes.most_common(1)[0][0])

    # Per-field entropy
    field_samples: Dict[str, List[str]] = {}
    for seq in all_tag_seqs:
        for tag_idx in seq:
            tag_str = IDX2TAG.get(tag_idx, "O")
            if tag_str == "O":
                continue
            field = tag_str.split("-", 1)[-1] if "-" in tag_str else tag_str
            field_samples.setdefault(field, []).append(tag_str)

    entropies: Dict[str, float] = {}
    for field, samples in field_samples.items():
        counts = Counter(samples)
        total  = len(samples)
        H = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                H -= p * math.log(p + 1e-12)
        entropies[field] = round(H, 6)

    return best_tags, entropies


def compute_confidence(entropies: Dict[str, float]) -> Dict[str, float]:
    """confidence = 1 − H / log(num_classes)"""
    max_H = math.log(NUM_TAGS + 1e-12)
    return {
        field: round(max(0.0, 1.0 - H / max_H), 4)
        for field, H in entropies.items()
    }


def overall_entropy(entropies: Dict[str, float]) -> float:
    if not entropies:
        return 0.0
    return round(sum(entropies.values()) / len(entropies), 6)


def should_route_to_human(entropies: Dict[str, float]) -> bool:
    return any(H > ENTROPY_THRESHOLD for H in entropies.values())


# ═══════════════════════════════════════════════════════════════
#  Decision Engine (Layer 5)
# ═══════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Rule-based interpreter: entity record + confidence → workflow action.
    """

    def decide(
        self,
        record: Dict[str, str],
        confidences: Dict[str, float],
        entropies: Dict[str, float],
        raw_text: str = "",
        validation_status: str = "Unknown",
    ) -> Dict:
        urgency  = self._classify_urgency(raw_text, confidences)
        action   = self._select_action(record)
        routing  = self._routing(confidences, entropies)

        return {
            "record":            record,
            "confidence":        {**confidences},
            "urgency":           urgency,
            "action_triggered":  action,
            "routing":           routing,
            "validation_status": validation_status,
        }

    def _classify_urgency(self, text: str, confidences: Dict) -> str:
        """
        3-tier urgency using the full emoji vocabulary:
          HIGH   → any ANGER/ALERT emoji or urgency keyword
          MEDIUM → SADNESS/SARCASM/MILD emojis, or low avg confidence
          LOW    → POSITIVE/NEUTRAL emojis only, high confidence
        """
        # Tier 1: High urgency — anger emojis + text keywords + alerts
        for kw in HIGH_URGENCY_KEYWORDS:
            if kw in text:
                return "high"

        # Tier 2: Medium urgency — sadness, sarcasm, mild frustration emojis
        for em in MEDIUM_URGENCY_EMOJIS:
            if em in text:
                return "medium"

        # Tier 3: Confidence-based fallback
        avg_conf = sum(confidences.values()) / max(len(confidences), 1)
        if avg_conf < CONFIDENCE_LOW_THRESHOLD:
            return "medium"    # uncertain prediction → don't auto-close

        # Positive emojis → low urgency (user is polite / issue already resolved)
        for em in LOW_URGENCY_EMOJIS:
            if em in text:
                return "low"

        return "medium"        # default: treat unknown as medium

    def _select_action(self, record: Dict) -> str:
        # Check for phrase matches in ISSUE_TYPE (E-commerce)
        issue = record.get("ISSUE_TYPE", "").lower()
        if issue:
            if any(k in issue for k in ["delivered", "missing", "where"]): return "zendesk_ticket"
            if any(k in issue for k in ["payment", "charge", "failed"]):  return "salesforce_case"
            if any(k in issue for k in ["wrong", "incorrect"]):           return "zendesk_ticket"
            if any(k in issue for k in ["damaged", "broken", "smashed"]) : return "zendesk_ticket"
            if any(k in issue for k in ["return", "back"]):               return "salesforce_case"
        
        # Check for phrase matches in EVENT (Aviation)
        event = record.get("EVENT", "").lower()
        if event:
            if any(k in event for k in ["delay", "late", "pushed"]): return "notify_passenger_slack"
            if any(k in event for k in ["cancel"]):                 return "notify_passenger_slack"
            if any(k in event for k in ["divert", "reroute"]):       return "notify_passenger_slack"
            if any(k in event for k in ["bag", "luggage", "lost"]):  return "zendesk_ticket"
            
        if "ORDER_ID" in record:  return "zendesk_ticket"
        if "FLIGHT_ID" in record: return "notify_passenger_slack"
        return "log_to_postgres"

    def _routing(self, confidences: Dict, entropies: Dict) -> str:
        if should_route_to_human(entropies):
            return "human_review"
        avg_conf = sum(confidences.values()) / max(len(confidences), 1)
        if avg_conf < CONFIDENCE_LOW_THRESHOLD:
            return "human_review"
        return "automated"
