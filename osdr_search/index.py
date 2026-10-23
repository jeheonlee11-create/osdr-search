"""Embed every dataset and serve nearest-neighbour search over the vectors.

The index is a plain numpy matrix of L2-normalised vectors (641 x 384), so
cosine similarity is one matrix-vector product. FAISS/HNSW would be the next
step if the corpus grew by a few orders of magnitude.
"""

import json
import sys
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

from .fetch import DATASETS_PATH, DATA_DIR

MODEL_NAME = "BAAI/bge-small-en-v1.5"
INDEX_PATH = DATA_DIR / "index.npz"


def doc_text(d):
    """The passage that gets embedded for one dataset."""
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


def _normalise(m):
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build():
    datasets = json.loads(DATASETS_PATH.read_text())
    model = TextEmbedding(MODEL_NAME)
    texts = [doc_text(d) for d in datasets]
    vectors = _normalise(np.array(list(model.embed(texts, batch_size=32)), dtype=np.float32))
    np.savez(INDEX_PATH, vectors=vectors, accessions=np.array([d["accession"] for d in datasets]))
    print(f"embedded {len(texts)} datasets -> {INDEX_PATH} {vectors.shape}", file=sys.stderr)


class VectorIndex:
    def __init__(self):
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"{INDEX_PATH} missing; run `python -m osdr_search.index`")
        z = np.load(INDEX_PATH)
        self.vectors = z["vectors"]
        self.accessions = [str(a) for a in z["accessions"]]
        self.model = TextEmbedding(MODEL_NAME)

    def search(self, query, k=50):
        """Return [(accession, cosine score)] for the k nearest datasets."""
        q = np.array(list(self.model.query_embed(query))[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self.accessions[i], float(scores[i])) for i in top]


if __name__ == "__main__":
    build()
