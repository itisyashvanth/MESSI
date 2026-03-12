
"""Tests for Layer 3: ILP Symbolic Validator"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from layers.layer3_ilp.constraints import (
    ConstraintValidator, extract_spans_from_bio, check_cardinality
)
from layers.layer3_ilp.solver import ILPValidator, validate_prediction


class TestConstraintValidator:
    def setup_method(self):
        self.v = ConstraintValidator()

    def test_valid_flight_id(self):
        assert self.v.is_valid("FLIGHT_ID", "UA123") is True
        assert self.v.is_valid("FLIGHT_ID", "AA9999") is True

    def test_invalid_flight_id(self):
        assert self.v.is_valid("FLIGHT_ID", "123") is False
        assert self.v.is_valid("FLIGHT_ID", "TOOLONG12345") is False

    def test_valid_order_id(self):
        assert self.v.is_valid("ORDER_ID", "782") is True
        assert self.v.is_valid("ORDER_ID", "0001") is True

    def test_invalid_order_id(self):
        assert self.v.is_valid("ORDER_ID", "abc") is False
        assert self.v.is_valid("ORDER_ID", "#782") is False

    def test_valid_issue_type(self):
        assert self.v.is_valid("ISSUE_TYPE", "not_delivered") is True
        assert self.v.is_valid("ISSUE_TYPE", "payment_failed") is True

    def test_invalid_issue_type(self):
        assert self.v.is_valid("ISSUE_TYPE", "random_garbage") is False

    def test_no_constraint_passes(self):
        # Fields without rules should always pass
        assert self.v.is_valid("UNKNOWN_FIELD", "anything") is True


class TestBIOToSpans:
    def test_simple_span(self):
        tokens = ["order", "782", "not", "delivered"]
        tags   = ["O", "B-ORDER_ID", "O", "B-ISSUE_TYPE"]
        spans  = extract_spans_from_bio(tokens, tags)
        assert len(spans) == 2
        assert spans[0]["field"] == "ORDER_ID"
        assert spans[0]["text"]  == "782"
        assert spans[1]["field"] == "ISSUE_TYPE"
        assert spans[1]["text"]  == "delivered"

    def test_multi_token_span(self):
        tokens = ["not", "delivered", "at", "all"]
        tags   = ["B-ISSUE_TYPE", "I-ISSUE_TYPE", "O", "O"]
        spans  = extract_spans_from_bio(tokens, tags)
        assert len(spans) == 1
        assert spans[0]["text"] == "not delivered"

    def test_empty_tags(self):
        tokens = ["hello", "world"]
        tags   = ["O", "O"]
        spans  = extract_spans_from_bio(tokens, tags)
        assert spans == []


class TestILPSolver:
    def setup_method(self):
        self.solver = ILPValidator()

    def test_passes_valid_spans(self):
        tokens = ["order", "782", "not", "delivered", "😠"]
        tags   = ["O", "B-ORDER_ID", "O", "B-ISSUE_TYPE", "O"]
        result = validate_prediction(tokens, tags, validator=self.solver)
        assert result["validation_status"] == "Passed ILP Constraints"
        assert "ORDER_ID" in result["record"]

    def test_invalid_flight_fails(self):
        tokens = ["flight", "123", "cancelled"]
        tags   = ["O", "B-FLIGHT_ID", "O"]
        result = validate_prediction(tokens, tags, validator=self.solver)
        # "123" does not match [A-Z]{2}\d{3,4} → should fail or not be selected
        assert result["record"].get("FLIGHT_ID") is None \
               or result["validation_status"] == "Failed ILP Constraints"

    def test_valid_flight_passes(self):
        tokens = ["flight", "UA123", "was", "delayed"]
        tags   = ["O", "B-FLIGHT_ID", "O", "B-EVENT"]
        result = validate_prediction(tokens, tags, validator=self.solver)
        assert result["record"].get("FLIGHT_ID") == "UA123"

    def test_empty_input(self):
        result = validate_prediction(["hello"], ["O"], validator=self.solver)
        assert result["record"] == {}
