"""BM25 and reciprocal rank fusion, on tiny in-memory corpora."""

import pytest

from osdr_search.bm25 import BM25, tokenize
from osdr_search.fusion import RRF_K, rrf


def test_tokenize_lowercases_splits_and_drops_stopwords():
    assert tokenize("How does RR-6 affect the Mouse liver?") == ["rr", "6", "affect", "mouse", "liver"]


def test_bm25_prefers_rarer_and_repeated_terms():
    index = BM25(["a", "b", "c"], [
        "mouse liver transcriptome",
        "mouse kidney transcriptome",
        "mouse liver liver proteome",
    ])
    hits = index.search("liver")
    assert [doc for doc, _ in hits] == ["c", "a"]  # "b" has no query term, so it is not returned
    assert hits[0][1] > hits[1][1] > 0
    # "mouse" is in every document: it scores, but less than a term that discriminates
    assert dict(index.search("mouse kidney"))["b"] > dict(index.search("mouse kidney"))["a"]


def test_bm25_length_normalisation():
    index = BM25(["short", "long"], ["soleus muscle", "soleus " + "filler " * 40])
    assert [doc for doc, _ in index.search("soleus")] == ["short", "long"]


def test_bm25_unknown_terms_return_nothing():
    assert BM25(["a"], ["mouse liver"]).search("zebrafish") == []


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    fused = dict(rrf({"x": ["a", "b"], "y": ["b", "c"]}))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["c"] == pytest.approx(1 / (RRF_K + 2))


def test_rrf_rewards_agreement_over_one_strong_vote():
    # "b" is second in both lists; "a" is first in one and absent from the other
    order = [doc for doc, _ in rrf({"x": ["a", "b"], "y": ["c", "b"]})]
    assert order[0] == "b"


def test_rrf_weights_and_deterministic_ties():
    assert [d for d, _ in rrf({"x": ["a"], "y": ["b"]})] == ["a", "b"]  # tie -> by id
    assert [d for d, _ in rrf({"x": ["a"], "y": ["b"]}, weights={"y": 2.0})] == ["b", "a"]
