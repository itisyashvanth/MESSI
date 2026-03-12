"""Tests for Layer 4: Uncertainty Estimation (MC Dropout & Entropy)"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from layers.layer4_uncertainty.entropy import predictive_entropy, field_entropy_summary
from layers.layer4_uncertainty.mc_dropout import (
    compute_confidence, overall_entropy, should_route_to_human
)
from config import IDX2TAG, ENTROPY_THRESHOLD


class TestPredictiveEntropy:
    def test_zero_entropy_when_all_agree(self):
        """If all T passes predict same tag, entropy should be 0."""
        counts = {"B-ORDER_ID": 10}
        H = predictive_entropy(counts, T=10)
        assert abs(H) < 1e-6

    def test_max_entropy_uniform(self):
        """Uniform distribution over C tags has maximum entropy log(C)."""
        C = 4
        counts = {f"tag_{i}": 10 for i in range(C)}
        H = predictive_entropy(counts, T=10 * C)
        expected = math.log(C)
        assert abs(H - expected) < 0.01

    def test_entropy_non_negative(self):
        counts = {"A": 7, "B": 3}
        H = predictive_entropy(counts, T=10)
        assert H >= 0.0

    def test_entropy_bounds(self):
        for n_tags in [2, 5, 10]:
            counts = {f"tag_{i}": 1 for i in range(n_tags)}
            H = predictive_entropy(counts, T=n_tags)
            assert 0 <= H <= math.log(n_tags) + 1e-6


class TestConfidenceComputation:
    def test_high_confidence_low_entropy(self):
        entropies = {"ORDER_ID": 0.01}
        confidences = compute_confidence(entropies)
        assert confidences["ORDER_ID"] > 0.9

    def test_low_confidence_high_entropy(self):
        entropies = {"ORDER_ID": 3.0}
        confidences = compute_confidence(entropies)
        assert confidences["ORDER_ID"] < 0.5

    def test_confidence_in_unit_range(self):
        entropies = {"ORDER_ID": 0.8, "ISSUE_TYPE": 0.3}
        confidences = compute_confidence(entropies)
        for v in confidences.values():
            assert 0.0 <= v <= 1.0

    def test_overall_entropy_mean(self):
        entropies = {"A": 0.4, "B": 0.6}
        oe = overall_entropy(entropies)
        assert abs(oe - 0.5) < 1e-6

    def test_human_routing_trigger(self):
        entropies = {"ORDER_ID": ENTROPY_THRESHOLD + 0.1}
        assert should_route_to_human(entropies) is True

    def test_no_human_routing_below_threshold(self):
        entropies = {"ORDER_ID": ENTROPY_THRESHOLD - 0.1}
        assert should_route_to_human(entropies) is False
