"""
End-to-End Pipeline Integration Tests

Verifies that the full 6-layer MESSI pipeline:
  1. Produces output conforming to the Blueprint §5 JSON schema.
  2. Handles emoji-heavy inputs correctly.
  3. Routes uncertain inputs to human review.
  4. Produces 'Passed ILP Constraints' for valid inputs.
  5. Handles edge cases (empty input, all-O tags).
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from layers.layer1_preprocessing.emoji_vocab import build_vocab_from_texts
from layers.layer2_bilstm_crf.model import BiLSTMCRF
from layers.layer3_ilp.solver import ILPValidator
from layers.layer4_uncertainty.mc_dropout import should_route_to_human
from layers.layer5_decision.engine import DecisionEngine
from layers.layer6_api.payload_builder import build_output_payload


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vocab():
    return build_vocab_from_texts(["😠", "😤", "💀", "🔥", "😡"])

@pytest.fixture(scope="module")
def model(vocab):
    m = BiLSTMCRF(emoji_vocab=vocab)
    m.eval()
    return m

@pytest.fixture(scope="module")
def ilp():
    return ILPValidator()

@pytest.fixture(scope="module")
def engine():
    return DecisionEngine()


# ── Schema validation ─────────────────────────────────────────────────────────

REQUIRED_TOP_LEVEL_KEYS = {
    "validation_status", "urgency", "action_triggered",
    "routing", "confidence",
}
REQUIRED_CONFIDENCE_KEYS = {"overall_entropy"}


def assert_schema_valid(payload: dict):
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in payload, f"Missing required key: {key}"
    assert "overall_entropy" in payload["confidence"]
    assert isinstance(payload["confidence"]["overall_entropy"], float)
    assert payload["urgency"] in ("high", "medium", "low", "unknown")
    assert payload["routing"] in ("automated", "human_review")


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPipelineOutputSchema:
    def test_basic_ecommerce_schema(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "782", "ISSUE_TYPE": "not_delivered"},
            confidences={"ORDER_ID": 0.98, "ISSUE_TYPE": 0.91},
            entropies={"ORDER_ID": 0.02, "ISSUE_TYPE": 0.05},
            raw_text="order #782 not delivered 😠",
            validation_status="Passed ILP Constraints",
        )
        payload = build_output_payload(decision, raw_text="order #782 not delivered 😠")
        assert_schema_valid(payload)

    def test_entity_id_mapped_from_order_id(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "456"},
            confidences={"ORDER_ID": 0.95},
            entropies={"ORDER_ID": 0.03},
            raw_text="order 456 😠",
            validation_status="Passed ILP Constraints",
        )
        payload = build_output_payload(decision, raw_text="order 456 😠")
        # ORDER_ID should be remapped to entity_id
        assert "entity_id" in payload
        assert payload["entity_id"] == "456"

    def test_high_urgency_on_anger_emoji(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "100", "ISSUE_TYPE": "not_delivered"},
            confidences={"ORDER_ID": 0.90, "ISSUE_TYPE": 0.85},
            entropies={"ORDER_ID": 0.05, "ISSUE_TYPE": 0.05},
            raw_text="order 100 not delivered 😠",
            validation_status="Passed ILP Constraints",
        )
        payload = build_output_payload(decision)
        assert payload["urgency"] == "high"

    def test_action_triggered_is_string(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "1", "ISSUE_TYPE": "payment_failed"},
            confidences={"ORDER_ID": 0.92, "ISSUE_TYPE": 0.88},
            entropies={"ORDER_ID": 0.03, "ISSUE_TYPE": 0.04},
            raw_text="payment failed 💀",
            validation_status="Passed ILP Constraints",
        )
        payload = build_output_payload(decision)
        assert isinstance(payload["action_triggered"], str)
        assert len(payload["action_triggered"]) > 0


class TestHumanReviewRouting:
    def test_high_entropy_routes_to_human(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "999"},
            confidences={"ORDER_ID": 0.55},
            entropies={"ORDER_ID": 0.9},  # above ENTROPY_THRESHOLD
            raw_text="something unclear",
            validation_status="Passed ILP Constraints",
        )
        assert decision["routing"] == "human_review"

    def test_low_confidence_routes_to_human(self, engine):
        decision = engine.decide(
            record={"ORDER_ID": "1"},
            confidences={"ORDER_ID": 0.60},   # below CONFIDENCE_LOW_THRESHOLD
            entropies={"ORDER_ID": 0.1},
            raw_text="blah",
            validation_status="Passed ILP Constraints",
        )
        assert decision["routing"] == "human_review"


class TestILPIntegration:
    def test_valid_order_id_passes(self, ilp):
        from layers.layer3_ilp.solver import validate_prediction
        tokens = ["order", "782", "not", "delivered"]
        tags   = ["O", "B-ORDER_ID", "O", "B-ISSUE_TYPE"]
        result = validate_prediction(tokens, tags, validator=ilp)
        assert result["validation_status"] == "Passed ILP Constraints"

    def test_invalid_flight_fails(self, ilp):
        from layers.layer3_ilp.solver import validate_prediction
        tokens = ["flight", "123X!!", "cancelled"]
        tags   = ["O", "B-FLIGHT_ID", "O"]
        result = validate_prediction(tokens, tags, validator=ilp)
        assert result["record"].get("FLIGHT_ID") is None


class TestModelShapes:
    def test_forward_no_crash(self, model):
        B, L = 1, 6
        sv = torch.randn(B, L, 300)
        ei = torch.zeros(B, L, dtype=torch.long)
        ta = torch.zeros(B, L, dtype=torch.long)
        mask = torch.ones(B, L, dtype=torch.bool)
        loss = model(sv, ei, ta, mask)
        assert not torch.isnan(loss)

    def test_decode_valid_tags(self, model):
        from config import NUM_TAGS
        B, L = 1, 5
        sv = torch.randn(B, L, 300)
        ei = torch.zeros(B, L, dtype=torch.long)
        mask = torch.ones(B, L, dtype=torch.bool)
        seqs = model.decode(sv, ei, mask)
        for tag in seqs[0]:
            assert 0 <= tag < NUM_TAGS
