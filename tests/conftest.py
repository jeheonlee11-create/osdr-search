"""Shared fakes: a SearchEngine built from in-memory data, with no data/ files, model or network."""

import pytest

from osdr_search.schema import QUERY_FIELDS, Schema
from osdr_search.search import SearchEngine


class FakeIndex:
    """Stands in for VectorIndex: fixed cosine scores instead of a fastembed model."""

    def __init__(self, scores):
        self.scores = scores

    def search(self, text, k=10):
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:k]


def dataset(acc, title=None, **fields):
    d = {"accession": acc, "title": title or acc, "description": "", "organism": [], "material_type": [],
         "factors": [], "assay_technology": [], "assay_measurement": [], "mission": "", "project_type": ""}
    d.update(fields)
    return d


def sample_row(acc, **fields):
    r = {"id.accession": acc}
    for field, value in fields.items():
        r[QUERY_FIELDS[field]] = value
    return r


def make_engine(datasets, rows, scores, retriever="semantic", translator=None):
    return SearchEngine(use_llm=False, retriever=retriever, datasets=datasets, index=FakeIndex(scores),
                        schema=Schema(rows=rows), translator=translator)


@pytest.fixture
def engine_factory():
    return make_engine
