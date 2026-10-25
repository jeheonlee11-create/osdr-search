"""Ranking metrics with binary relevance.

`ranked` is the list of result ids as shown (best first); `relevant` is the
set of ids that count as correct; `n_relevant` is how many correct results a
perfect ranking could show (it can be smaller than len(relevant) when results
are collapsed, see run_eval.py).
"""

import math


def recall_at_k(ranked, relevant, k, n_relevant=None):
    """Capped recall (BEIR's R_cap@k): hits in the top k / min(k, number of relevant results).

    Plain recall@5 cannot exceed 5/30 for a query with 30 correct datasets; the
    cap makes 1.0 mean "the top k is as full of correct results as it can be".
    """
    n = len(relevant) if n_relevant is None else n_relevant
    if n == 0:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / min(k, n)


def reciprocal_rank(ranked, relevant, k=10):
    """1 / rank of the first correct result in the top k, else 0. Averaged over queries this is MRR@k."""
    for i, r in enumerate(ranked[:k], start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k=10, n_relevant=None):
    """Normalised discounted cumulative gain: rewards correct results more the higher they appear."""
    n = len(relevant) if n_relevant is None else n_relevant
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, n) + 1))
    return dcg / ideal if ideal else 0.0
