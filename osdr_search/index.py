"""Embed every dataset and serve nearest-neighbour search over the vectors.

The index is a plain numpy matrix of L2-normalised vectors (641 x 384), so
cosine similarity is one matrix-vector product. FAISS/HNSW would be the next
step if the corpus grew by a few orders of magnitude.

data/datasets.json is tracked in git but data/index.npz is not, so the index
stores a fingerprint of the corpus it was built from and refuses to load
against a different one (instead of failing later with a KeyError).
"""

import hashlib
import json
import sys

import numpy as np

from .fetch import DATA_DIR, DATASETS_PATH

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_PATH = DATA_DIR / "index.npz"
REBUILD_HINT = "rebuild it with `python -m osdr_search.index`"


class CorpusMismatchError(RuntimeError):
    """data/index.npz was built from a different data/datasets.json (or embedding model)."""


def doc_text(d):
    """The passage that gets embedded (and indexed by BM25) for one dataset."""
    parts = [d["title"]]
    if d["organism"]:
        parts.append("Organism: " + ", ".join(d["organism"]))
    if d["material_type"]:
        parts.append("Tissue or material: " + ", ".join(d["material_type"]))
    if d["factors"]:
        parts.append("Experimental factors: " + ", ".join(d["factors"]))
    assays = sorted(set(d["assay_technology"] + d["assay_measurement"]))
    if assays:
        parts.append("Assays: " + ", ".join(assays))
    if d["mission"]:
        parts.append("Mission: " + d["mission"])
    if d["project_type"]:
        parts.append(d["project_type"])
    if d["description"]:
        parts.append(d["description"])
    return ". ".join(p.rstrip(".") for p in parts if p)


def corpus_fingerprint(datasets, model_name=MODEL_NAME):
    """sha256 over the embedding model and every (accession, passage) pair, in order."""
    h = hashlib.sha256(model_name.encode())
    for d in datasets:
        h.update(b"\0" + d["accession"].encode() + b"\0" + doc_text(d).encode())
    return h.hexdigest()


def check_fingerprint(index, datasets):
    """Raise CorpusMismatchError unless `index` was built from exactly these datasets."""
    if index.fingerprint is None:
        raise CorpusMismatchError(f"{INDEX_PATH} has no corpus fingerprint (built by an older version); "
                                  + REBUILD_HINT)
    if index.fingerprint != corpus_fingerprint(datasets):
        missing = sorted(set(index.accessions) - {d["accession"] for d in datasets})
        detail = f" ({len(missing)} indexed accessions are not in datasets.json, e.g. {missing[:3]})" if missing else ""
        raise CorpusMismatchError(f"{INDEX_PATH} was built from a different {DATASETS_PATH.name} or embedding "
                                  f"model{detail}; {REBUILD_HINT}")


def _embedder():
    from fastembed import TextEmbedding  # imported lazily: tests never load (or download) the model

    return TextEmbedding(MODEL_NAME)


def _normalise(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build(path=INDEX_PATH):
    datasets = json.loads(DATASETS_PATH.read_text())
    texts = [doc_text(d) for d in datasets]
    vectors = _normalise(np.array(list(_embedder().embed(texts, batch_size=32)), dtype=np.float32))
    np.savez(path, vectors=vectors, accessions=np.array([d["accession"] for d in datasets]),
             fingerprint=np.array(corpus_fingerprint(datasets)))
    print(f"embedded {len(texts)} datasets -> {path} {vectors.shape}", file=sys.stderr)


class VectorIndex:
    def __init__(self, path=INDEX_PATH, embedder=None):
        """embedder: anything with fastembed's query_embed(text); loaded on first query when omitted."""
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `python -m osdr_search.index`")
        z = np.load(path)
        self.vectors = z["vectors"]
        self.accessions = [str(a) for a in z["accessions"]]
        self.fingerprint = str(z["fingerprint"]) if "fingerprint" in z.files else None
        self._embedder = embedder

    @property
    def model(self):
        if self._embedder is None:
            self._embedder = _embedder()
        return self._embedder

    def search(self, query, k=50):
        """Return [(accession, cosine score)] for the k nearest datasets."""
        q = np.array(list(self.model.query_embed(query))[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.accessions[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    build()
