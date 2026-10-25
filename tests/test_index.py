"""Corpus fingerprint: an index built from one datasets.json must not load against another."""

import numpy as np
import pytest
from conftest import dataset

from osdr_search.index import CorpusMismatchError, VectorIndex, check_fingerprint, corpus_fingerprint
from osdr_search.schema import Schema
from osdr_search.search import SearchEngine


class FakeEmbedder:
    def query_embed(self, text):
        yield np.array([1.0, 0.0], dtype=np.float32)


DATASETS = [dataset("OSD-1", "mouse liver"), dataset("OSD-2", "rat soleus")]


def write_index(path, datasets, fingerprint=True):
    extra = {"fingerprint": np.array(corpus_fingerprint(datasets))} if fingerprint else {}
    np.savez(path, vectors=np.eye(len(datasets), 2, dtype=np.float32),
             accessions=np.array([d["accession"] for d in datasets]), **extra)
    return path


def test_fingerprint_changes_with_any_passage_or_the_model():
    base = corpus_fingerprint(DATASETS)
    assert base == corpus_fingerprint([dict(d) for d in DATASETS])
    assert base != corpus_fingerprint([DATASETS[0], dataset("OSD-2", "rat soleus muscle")])
    assert base != corpus_fingerprint(DATASETS[:1])
    assert base != corpus_fingerprint(DATASETS, model_name="another-model")


def test_matching_index_loads_and_searches(tmp_path):
    index = VectorIndex(write_index(tmp_path / "index.npz", DATASETS), embedder=FakeEmbedder())
    check_fingerprint(index, DATASETS)
    assert index.search("anything", k=1) == [("OSD-1", 1.0)]


def test_index_from_another_corpus_is_rejected_with_rebuild_hint(tmp_path):
    index = VectorIndex(write_index(tmp_path / "index.npz", DATASETS + [dataset("OSD-9")]), embedder=FakeEmbedder())
    with pytest.raises(CorpusMismatchError, match=r"OSD-9.*python -m osdr_search.index"):
        check_fingerprint(index, DATASETS)


def test_index_without_fingerprint_is_rejected(tmp_path):
    index = VectorIndex(write_index(tmp_path / "index.npz", DATASETS, fingerprint=False), embedder=FakeEmbedder())
    with pytest.raises(CorpusMismatchError, match="no corpus fingerprint"):
        check_fingerprint(index, DATASETS)


def test_search_engine_checks_at_startup(tmp_path):
    stale = VectorIndex(write_index(tmp_path / "index.npz", DATASETS[:1]), embedder=FakeEmbedder())
    with pytest.raises(CorpusMismatchError):
        SearchEngine(use_llm=False, datasets=DATASETS, index=stale, schema=Schema(rows=[]))
