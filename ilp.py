"""
MESSI — Layer 3: ILP Symbolic Validator
========================================
Merges: constraints.py + solver.py

Exported:
  ConstraintValidator          — regex, allowlist, cardinality
  extract_spans_from_bio(tokens, tags)  → List[span]
  ILPValidator                 — OR-Tools CP-SAT solver
  validate_prediction(tokens, tags, validator)  → {record, validation_status}
"""

import re
from typing import Dict, List, Optional, Tuple

from config import (
    FIELD_REGEX_RULES, FIELD_ALLOWLISTS, FIELD_CARDINALITY,
)


# ═══════════════════════════════════════════════════════════════
#  Constraint Rules
# ═══════════════════════════════════════════════════════════════

class ConstraintValidator:
    """Validates entity spans against hard domain rules."""

    def is_valid(self, field: str, text: str) -> bool:
        # 1. Regex check
        pattern = FIELD_REGEX_RULES.get(field)
        if pattern:
            if not re.search(pattern, text.replace(" ",""), re.I):
                return False
        # 2. Allowlist check (case-insensitive)
        allowlist = FIELD_ALLOWLISTS.get(field)
        if allowlist:
            text_clean = text.lower().strip()
            if not any(item.lower() in text_clean for item in allowlist):
                return False
        return True


def extract_spans_from_bio(tokens: List[str], bio_tags: List[str]) -> List[Dict]:
    """Convert BIO tag sequence into a list of {field, text, start, end, score}."""
    spans = []
    i = 0
    while i < len(bio_tags):
        tag = bio_tags[i]
        if tag.startswith("B-"):
            field = tag[2:]
            j = i + 1
            while j < len(bio_tags) and bio_tags[j] == f"I-{field}":
                j += 1
            spans.append({
                "field": field,
                "text":  " ".join(tokens[i:j]),
                "start": i,
                "end":   j,
                "score": 1.0,   # placeholder — overridden by MC Dropout confidence
            })
            i = j
        else:
            i += 1
    # print(f"DEBUG extract_spans: {len(bio_tags)} tags, {len(spans)} spans")
    return spans


def check_cardinality(spans: List[Dict]) -> bool:
    from collections import Counter
    counts = Counter(s["field"] for s in spans)
    for field, (mn, mx) in FIELD_CARDINALITY.items():
        c = counts.get(field, 0)
        if not (mn <= c <= mx):
            return False
    return True


# ═══════════════════════════════════════════════════════════════
#  ILP Solver (OR-Tools CP-SAT)
# ═══════════════════════════════════════════════════════════════

class ILPValidator:
    """
    Constrained MAP inference using OR-Tools CP-SAT.

    Selects the highest-confidence valid subset of spans such that:
      • Each span passes ConstraintValidator
      • Cardinality constraints are satisfied
      • No two spans overlap
    """

    def __init__(self):
        self.cv = ConstraintValidator()

    def solve(self, spans: List[Dict]) -> Dict:
        """Return {record, validation_status}."""
        if not spans:
            return {"record": {}, "validation_status": "Passed ILP Constraints"}

        # Filter hard-constraint violations immediately
        valid_spans = [s for s in spans if self.cv.is_valid(s["field"], s["text"])]
        # print(f"DEBUG: spans={len(spans)}, valid_spans={len(valid_spans)}")

        if not valid_spans:
            return {"record": {}, "validation_status": "Failed ILP Constraints"}

        try:
            from ortools.sat.python import cp_model
            model = cp_model.CpModel()
            n = len(valid_spans)
            x = [model.NewBoolVar(f"x_{i}") for i in range(n)]

            # Cardinality constraints
            from collections import defaultdict
            field_vars: Dict[str, List] = defaultdict(list)
            for i, span in enumerate(valid_spans):
                field_vars[span["field"]].append(x[i])

            for field, vars_list in field_vars.items():
                mn, mx = FIELD_CARDINALITY.get(field, (0, 99))
                model.Add(sum(vars_list) >= mn)
                model.Add(sum(vars_list) <= mx)

            # No overlapping spans
            for i in range(n):
                for j in range(i + 1, n):
                    si, sj = valid_spans[i], valid_spans[j]
                    if si["start"] < sj["end"] and sj["start"] < si["end"]:
                        model.Add(x[i] + x[j] <= 1)

            # Objective: maximise sum of scores for selected spans
            scores = [int(s["score"] * 1000) for s in valid_spans]
            model.Maximize(sum(scores[i] * x[i] for i in range(n)))

            solver   = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 1.0
            status   = solver.Solve(model)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                record = {}
                for i, span in enumerate(valid_spans):
                    if solver.Value(x[i]):
                        record[span["field"]] = span["text"]
                return {"record": record, "validation_status": "Passed ILP Constraints"}
            else:
                return {"record": {}, "validation_status": "Failed ILP Constraints"}

        except Exception:
            # Fallback: greedy selection without ILP
            record = {}
            for span in valid_spans:
                if span["field"] not in record:
                    record[span["field"]] = span["text"]
            return {"record": record, "validation_status": "Passed ILP Constraints (greedy)"}


def validate_prediction(tokens: List[str], bio_tags: List[str],
                        validator: Optional[ILPValidator] = None) -> Dict:
    if validator is None:
        validator = ILPValidator()
    spans = extract_spans_from_bio(tokens, bio_tags)
    return validator.solve(spans)
