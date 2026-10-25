"""Okapi BM25 keyword ranking over the same passages that get embedded.

A small in-repo implementation (about 50 lines) instead of the rank_bm25
package: the corpus is 641 short documents, so an inverted index in plain
Python scores a query in well under a millisecond and adds no dependency.
"""

import math
import re
from collections import Counter, defaultdict

# Function words that carry no topical signal in these questions. Kept short on
# purpose: words like "space", "flight" or "mice" are informative here.
STOPWORDS = frozenset("""
a an and are as at be by can do does for from has have how i in into is it its of on or
that the their there these this to was were what when which who why with
""".split())

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lower-case alphanumeric tokens with stopwords removed ("RR-6" -> ["rr", "6"])."""
    return [w for w in _WORD.findall(str(text).lower()) if w not in STOPWORDS]


class BM25:
    """BM25 with the usual defaults (k1=1.5, b=0.75) and a non-negative idf."""

    def __init__(self, ids, texts, k1=1.5, b=0.75):
        self.ids = list(ids)
        self.k1, self.b = k1, b
        docs = [tokenize(t) for t in texts]
        self.doc_len = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_len) / len(docs)) if docs else 0.0
        n = len(docs)
        # postings[term] -> [(doc index, term frequency)]
        self.postings = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query):
        """{doc index: score} for every document sharing at least one query term."""
        out = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query, k=50):
        """[(id, score)] for the k best-scoring documents; documents scoring 0 are left out."""
        scored = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.ids[i], s) for i, s in scored]
