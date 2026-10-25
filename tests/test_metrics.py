"""Evaluation metrics on hand-computed examples."""

import math

import pytest

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_is_capped_by_k():
    relevant = {f"r{i}" for i in range(30)}
    ranked = ["r0", "x", "r1", "r2", "r3"]
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(4 / 5)  # not 4 / 30
    assert recall_at_k(["x", "r0"], {"r0"}, 5) == 1.0
    assert recall_at_k(["x"], set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x"] * 10 + ["a"], {"a"}, k=10) == 0.0


def test_ndcg():
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
    # one relevant result at rank 2 instead of rank 1
    assert ndcg_at_k(["x", "a"], {"a"}) == pytest.approx((1 / math.log2(3)) / 1.0)
    # n_relevant lowers the ideal when collapsing makes some answers unreachable
    assert ndcg_at_k(["a"], {"a", "b"}, n_relevant=1) == pytest.approx(1.0)
