"""
MESSI — Consolidated Test Suite
All layer tests in one file.
Run: pytest tests/test_messi.py -v
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from preprocessing import (
    split_emoji_tokens, build_emoji_aware_nlp, tokenize,
    build_vocab_from_texts, get_emoji_index, CombinedEmbedding, EmbeddingExtractor,
)
from model import BiLSTMCRF, CRFLayer
from ilp import ConstraintValidator, extract_spans_from_bio, ILPValidator, validate_prediction
from uncertainty import (
    predictive_entropy, mc_dropout_predict, compute_confidence,
    overall_entropy, should_route_to_human, DecisionEngine,
)
from api import build_output_payload, dispatch_action
from config import NUM_TAGS, IDX2TAG, TAG2IDX, ENTROPY_THRESHOLD, EMBEDDING_DIM


# ─── Shared fixtures ──────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def vocab():
    return build_vocab_from_texts(["😠", "😡", "💀", "🔥", "😤"])

@pytest.fixture(scope="module")
def model(vocab):
    return BiLSTMCRF(emoji_vocab=vocab)

@pytest.fixture(scope="module")
def nlp():
    return build_emoji_aware_nlp()

@pytest.fixture(scope="module")
def ilp():
    return ILPValidator()

@pytest.fixture(scope="module")
def engine():
    return DecisionEngine()


# ─── Layer 1: Preprocessing ───────────────────────────────────────────────────
class TestPreprocessing:
    def test_emoji_split_adjacent(self):
        assert "😤" in split_emoji_tokens("delayed😤")

    def test_no_emoji_unchanged(self):
        assert split_emoji_tokens("hello world") == ["hello", "world"]

    def test_multi_emoji(self):
        t = split_emoji_tokens("fail💀🔥")
        assert "💀" in t and "🔥" in t

    def test_no_empty_tokens(self):
        assert all(t.strip() for t in split_emoji_tokens("  😠  "))

    def test_vocab_has_pad_unk(self, vocab):
        assert "<PAD>" in vocab and "<UNK>" in vocab
        assert vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1

    def test_known_emoji_lookup(self, vocab):
        assert get_emoji_index("😠", vocab) >= 2

    def test_unk_fallback(self, vocab):
        assert get_emoji_index("🛸", vocab) == vocab["<UNK>"]

    def test_tokenize_returns_list(self, nlp):
        result = tokenize("order #782 not delivered 😠", nlp)
        assert isinstance(result, list) and len(result) > 0

    def test_emoji_separate_token(self, nlp):
        assert "😠" in tokenize("notdelivered😠", nlp)

    def test_combined_embedding_shape(self, vocab):
        emb = CombinedEmbedding(vocab)
        sv  = torch.randn(1, 4, 300)
        ei  = torch.zeros(1, 4, dtype=torch.long)
        out = emb(sv, ei)
        assert out.shape == (1, 4, EMBEDDING_DIM)


# ─── Layer 2: BiLSTM-CRF ──────────────────────────────────────────────────────
class TestModel:
    def test_crf_nll_scalar(self):
        crf   = CRFLayer(NUM_TAGS)
        em    = torch.randn(2, 8, NUM_TAGS)
        tags  = torch.zeros(2, 8, dtype=torch.long)
        mask  = torch.ones(2, 8, dtype=torch.bool)
        loss  = crf.neg_log_likelihood(em, tags, mask)
        assert loss.shape == torch.Size([]) and not torch.isnan(loss)

    def test_viterbi_length(self):
        crf  = CRFLayer(NUM_TAGS)
        em   = torch.randn(2, 6, NUM_TAGS)
        mask = torch.ones(2, 6, dtype=torch.bool)
        seqs = crf.viterbi(em, mask)
        assert len(seqs) == 2 and all(len(s) == 6 for s in seqs)

    def test_viterbi_valid_indices(self):
        crf  = CRFLayer(NUM_TAGS)
        em   = torch.randn(1, 10, NUM_TAGS)
        mask = torch.ones(1, 10, dtype=torch.bool)
        for tag in crf.viterbi(em, mask)[0]:
            assert 0 <= tag < NUM_TAGS

    def test_bilstm_forward_loss(self, model):
        sv   = torch.randn(2, 7, 300)
        ei   = torch.zeros(2, 7, dtype=torch.long)
        tags = torch.zeros(2, 7, dtype=torch.long)
        mask = torch.ones(2, 7, dtype=torch.bool)
        loss = model(sv, ei, tags, mask)
        assert not torch.isnan(loss)

    def test_bilstm_decode_list(self, model):
        sv, ei = torch.randn(1, 5, 300), torch.zeros(1, 5, dtype=torch.long)
        mask   = torch.ones(1, 5, dtype=torch.bool)
        result = model.decode(sv, ei, mask)
        assert isinstance(result[0], list) and len(result[0]) == 5


# ─── Layer 3: ILP ─────────────────────────────────────────────────────────────
class TestILP:
    def test_valid_flight_id(self):
        cv = ConstraintValidator()
        assert cv.is_valid("FLIGHT_ID", "UA123") is True
        assert cv.is_valid("FLIGHT_ID", "123")   is False

    def test_valid_order_id(self):
        cv = ConstraintValidator()
        assert cv.is_valid("ORDER_ID", "782") is True
        assert cv.is_valid("ORDER_ID", "abc") is False

    def test_valid_issue_type(self):
        cv = ConstraintValidator()
        assert cv.is_valid("ISSUE_TYPE", "not_delivered")  is True
        assert cv.is_valid("ISSUE_TYPE", "random_garbage") is False

    def test_bio_to_spans(self):
        spans = extract_spans_from_bio(
            ["order", "782", "not", "delivered"],
            ["O", "B-ORDER_ID", "O", "B-ISSUE_TYPE"]
        )
        assert len(spans) == 2 and spans[0]["field"] == "ORDER_ID"

    def test_multi_token_span(self):
        spans = extract_spans_from_bio(
            ["not", "delivered", "at", "all"],
            ["B-ISSUE_TYPE", "I-ISSUE_TYPE", "O", "O"]
        )
        assert spans[0]["text"] == "not delivered"

    def test_ilp_valid_order(self, ilp):
        r = validate_prediction(["order","782","not","delivered","😠"],
                                ["O","B-ORDER_ID","O","B-ISSUE_TYPE","O"], validator=ilp)
        assert r["validation_status"] == "Passed ILP Constraints"

    def test_ilp_invalid_flight(self, ilp):
        r = validate_prediction(["flight","123X!!"],
                                ["O","B-FLIGHT_ID"], validator=ilp)
        assert r["record"].get("FLIGHT_ID") is None


# ─── Layer 4: Uncertainty ─────────────────────────────────────────────────────
class TestUncertainty:
    def test_zero_entropy_certain(self):
        assert abs(predictive_entropy({"A": 10}, T=10)) < 1e-6

    def test_entropy_non_negative(self):
        assert predictive_entropy({"A": 7, "B": 3}, T=10) >= 0

    def test_confidence_unit_range(self):
        for v in compute_confidence({"F1": 0.5, "F2": 0.1}).values():
            assert 0 <= v <= 1

    def test_overall_entropy_mean(self):
        assert abs(overall_entropy({"A": 0.4, "B": 0.6}) - 0.5) < 1e-6

    def test_human_routing_above_threshold(self):
        assert should_route_to_human({"F": ENTROPY_THRESHOLD + 0.01}) is True

    def test_no_routing_below_threshold(self):
        assert should_route_to_human({"F": ENTROPY_THRESHOLD - 0.01}) is False

    def test_mc_dropout_shape(self, model):
        sv, ei = torch.randn(1, 5, 300), torch.zeros(1, 5, dtype=torch.long)
        mask   = torch.ones(1, 5, dtype=torch.bool)
        tags, entropies = mc_dropout_predict(model, sv, ei, mask, T=3)
        assert isinstance(tags, list) and len(tags) == 5
        assert isinstance(entropies, dict)


# ─── Layers 5+6: Decision & Payload ──────────────────────────────────────────
class TestDecisionAndPayload:
    REQUIRED_KEYS = {"validation_status", "urgency", "action_triggered", "routing", "confidence"}

    def _decision(self, engine, **kw):
        defaults = dict(record={"ORDER_ID": "1", "ISSUE_TYPE": "not_delivered"},
                        confidences={"ORDER_ID": 0.98, "ISSUE_TYPE": 0.91},
                        entropies={"ORDER_ID": 0.02, "ISSUE_TYPE": 0.04},
                        raw_text="order 1 not delivered 😠",
                        validation_status="Passed ILP Constraints")
        defaults.update(kw)
        return engine.decide(**defaults)

    def test_schema_keys(self, engine):
        d = self._decision(engine)
        p = build_output_payload(d)
        for k in self.REQUIRED_KEYS:
            assert k in p

    def test_high_urgency_emoji(self, engine):
        d = self._decision(engine, raw_text="order 1 not delivered 😠")
        assert d["urgency"] == "high"

    def test_action_is_string(self, engine):
        d  = self._decision(engine)
        p  = build_output_payload(d)
        assert isinstance(p["action_triggered"], str)

    def test_entity_id_remapped(self, engine):
        d = self._decision(engine)
        p = build_output_payload(d)
        assert "entity_id" in p

    def test_human_review_high_entropy(self, engine):
        d = self._decision(engine, confidences={"ORDER_ID": 0.6},
                           entropies={"ORDER_ID": ENTROPY_THRESHOLD + 0.1})
        assert d["routing"] == "human_review"

    def test_dry_run_dispatch(self, engine):
        d = self._decision(engine)
        p = build_output_payload(d)
        p = dispatch_action(p, dry_run=True)
        assert p["api_response"]["status"] == "dry_run"
