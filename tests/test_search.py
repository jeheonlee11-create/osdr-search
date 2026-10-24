"""SearchEngine.rank with a fake index: the LLM's search_text/rationale must not show up as dropped filters."""

from osdr_search.llm import StructuredQuery
from osdr_search.schema import QUERY_FIELDS, Schema
from osdr_search.search import SearchEngine


class FakeIndex:
    def __init__(self, scores):
        self.scores = scores

    def search(self, text, k=10):
        return sorted(self.scores.items(), key=lambda kv: -kv[1])[:k]


def dataset(acc):
    return {"accession": acc, "title": acc, "organism": [], "material_type": [], "assay_technology": [],
            "mission": "", "project_type": ""}


def engine():
    rows = [
        {"id.accession": "OSD-1", QUERY_FIELDS["material_type"]: "Left Lobe of the Liver"},
        {"id.accession": "OSD-2", QUERY_FIELDS["material_type"]: "Heart"},
        {"id.accession": "OSD-3", QUERY_FIELDS["material_type"]: "Liver"},  # puts "Liver" in the vocabulary
    ]
    e = SearchEngine.__new__(SearchEngine)  # skip loading data/ files
    e.schema = Schema(rows=rows)
    e.index = FakeIndex({"OSD-2": 0.80, "OSD-1": 0.75})
    e.datasets = {a: dataset(a) for a in ("OSD-1", "OSD-2")}
    e.translator = None
    e.llm_enabled = False
    return e


def test_rank_reports_no_dropped_for_valid_llm_query():
    sq = StructuredQuery(material_type=["Liver"], search_text="liver", rationale="asked about liver")
    hits, clean, dropped = engine().rank("mouse liver?", sq, k=2)
    assert dropped == []
    assert clean == {"material_type": ["Liver"]}
    # OSD-1 ("Left Lobe of the Liver") satisfies the filter and overtakes OSD-2
    assert [h["accession"] for h in hits] == ["OSD-1", "OSD-2"]
    assert hits[0]["matched_filters"] == ["material_type"]


def test_rank_reports_invalid_llm_value_as_dropped():
    sq = StructuredQuery(material_type=["Liver", "Gills"], search_text="liver", rationale="")
    _, clean, dropped = engine().rank("q", sq, k=2)
    assert clean == {"material_type": ["Liver"]}
    assert dropped == ["material_type=Gills"]
