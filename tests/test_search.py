"""SearchEngine.rank with fake components: retrieval modes, fusion and filter re-ranking."""

from conftest import dataset, make_engine, sample_row

from osdr_search.llm import StructuredQuery


def liver_engine(retriever="semantic"):
    datasets = [dataset("OSD-1", "left lobe liver study"), dataset("OSD-2", "heart study")]
    rows = [
        sample_row("OSD-1", material_type="Left Lobe of the Liver"),
        sample_row("OSD-2", material_type="Heart"),
        sample_row("OSD-3", material_type="Liver"),  # puts "Liver" in the vocabulary
    ]
    return make_engine(datasets, rows, {"OSD-2": 0.80, "OSD-1": 0.75}, retriever=retriever)


def test_rank_reports_no_dropped_for_valid_llm_query():
    sq = StructuredQuery(material_type=["Liver"], search_text="liver", rationale="asked about liver")
    hits, clean, dropped = liver_engine().rank("mouse liver?", sq, k=2)
    assert dropped == []
    assert clean == {"material_type": ["Liver"]}
    # OSD-1 ("Left Lobe of the Liver") satisfies the filter and overtakes OSD-2
    assert [h["accession"] for h in hits] == ["OSD-1", "OSD-2"]
    assert hits[0]["matched_filters"] == ["material_type"]


def test_rank_reports_invalid_llm_value_as_dropped():
    sq = StructuredQuery(material_type=["Liver", "Gills"], search_text="liver", rationale="")
    _, clean, dropped = liver_engine().rank("q", sq, k=2)
    assert clean == {"material_type": ["Liver"]}
    assert dropped == ["material_type=Gills"]


def test_without_filters_semantic_order_is_kept():
    hits, _, _ = liver_engine().rank("anything", None, k=2)
    assert [h["accession"] for h in hits] == ["OSD-2", "OSD-1"]


def test_bm25_mode_uses_keywords_only():
    hits, _, _ = liver_engine("bm25").rank("liver", None, k=5)
    assert [h["accession"] for h in hits] == ["OSD-1"]
    assert hits[0]["ranks"] == {"bm25": 1}


def test_hybrid_fuses_both_rankers():
    # semantic puts OSD-2 first, BM25 only knows OSD-1 ("liver"): OSD-1 gets two votes
    hits, _, _ = liver_engine("hybrid").rank("liver", None, k=2)
    assert [h["accession"] for h in hits] == ["OSD-1", "OSD-2"]
    assert hits[0]["ranks"] == {"semantic": 2, "bm25": 1}
    assert hits[1]["ranks"] == {"semantic": 1, "bm25": None}


def test_filter_vote_is_on_the_rrf_scale():
    sq = StructuredQuery(material_type=["Liver"], search_text="liver", rationale="")
    plain, _, _ = liver_engine("hybrid").rank("liver", None, k=2)
    boosted, _, _ = liver_engine("hybrid").rank("liver", sq, k=2)
    # one extra ranker can add at most 1 / (60 + 1) to a fused score
    assert 0 < boosted[0]["score"] - plain[0]["score"] <= 1 / 61 + 1e-4  # scores are rounded to 5 places
