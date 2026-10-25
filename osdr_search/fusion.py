"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009).

Each ranker votes 1 / (k + rank) for every document it returns. Only ranks
are used, so BM25 scores (unbounded) and cosine similarities (squeezed into a
narrow band) can be combined without calibrating one against the other.
"""

from collections import defaultdict

RRF_K = 60  # the constant from the original paper; damps the influence of the very top ranks


def rrf(rankings, k=RRF_K, weights=None):
    """rankings: {name: [id, ...] best first} -> [(id, fused score)] best first.

    weights: optional {name: weight}; a missing name counts 1.0. Ties are broken
    by id so the output is deterministic.
    """
    fused = defaultdict(float)
    for name, ids in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, doc in enumerate(ids, start=1):
            fused[doc] += w / (k + rank)
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
