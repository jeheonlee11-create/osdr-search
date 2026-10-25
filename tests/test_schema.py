"""Filter validation and dataset matching, on a tiny in-memory sample table.

No network, no API key, no data/ files.
"""

import pytest

from osdr_search.schema import QUERY_FIELDS, Schema, phrase_in


def row(acc, **fields):
    r = {"id.accession": acc}
    for field, value in fields.items():
        r[QUERY_FIELDS[field]] = value
    return r


ROWS = [
    row("OSD-1", organism="Mus musculus", material_type="Left Lobe of the Liver",
        assay_measurement="transcription profiling"),
    row("OSD-2", organism="Mus musculus", material_type="Liver", assay_measurement="Transcription Profiling"),
    row("OSD-3", organism="Rattus norvegicus", material_type="Delivery Medium"),
    row("OSD-4", organism="Homo sapiens", material_type="Lunge Muscle"),
    row("OSD-5", organism="Mus musculus", material_type="Lung", spaceflight="Space Flight"),
    row("OSD-6", organism="Mus musculus", material_type="Left Lung", spaceflight="Pre-flight"),
    row("OSD-7", organism="Bacillus subtilis", assay_technology="protein microarray"),
]


@pytest.fixture(scope="module")
def schema():
    return Schema(rows=ROWS)


def test_vocab_built_from_injected_rows(schema):
    assert schema.sample_count == len(ROWS)
    assert "Left Lobe of the Liver" in schema.vocab["material_type"]


def test_structured_query_non_filter_fields_never_dropped(schema):
    from osdr_search.llm import StructuredQuery

    sq = StructuredQuery(material_type=["Liver"], search_text="mouse liver", rationale="liver was asked about")
    clean, dropped = schema.validate(sq.filters())
    assert clean == {"material_type": ["Liver"]}
    assert dropped == []


def test_invalid_value_and_unknown_field_still_dropped(schema):
    clean, dropped = schema.validate({"material_type": ["Liver", "Spleen"], "colour": ["blue"]})
    assert clean == {"material_type": ["Liver"]}
    assert "material_type=Spleen" in dropped
    assert "colour=* (unknown field)" in dropped


def test_liver_matches_left_lobe_of_the_liver(schema):
    matched, contradicted = schema.dataset_matches("OSD-1", {"material_type": ["Liver"]})
    assert matched == ["material_type"]
    assert contradicted == []


@pytest.mark.parametrize("needle,haystack", [("Liver", "Delivery Medium"), ("Lung", "Lunge Muscle")])
def test_substring_inside_a_word_is_not_a_match(needle, haystack):
    assert not phrase_in(needle, haystack)


def test_liver_does_not_match_delivery_dataset(schema):
    matched, contradicted = schema.dataset_matches("OSD-3", {"material_type": ["Liver"]})
    assert matched == []
    assert contradicted == ["material_type"]


def test_more_specific_filter_does_not_match_broader_value(schema):
    _, contradicted = schema.dataset_matches("OSD-2", {"material_type": ["Left Lobe of the Liver"]})
    assert contradicted == ["material_type"]


def test_real_contradiction_still_counts(schema):
    matched, contradicted = schema.dataset_matches(
        "OSD-5", {"organism": ["Mus musculus"], "material_type": ["Liver"]})
    assert matched == ["organism"]
    assert contradicted == ["material_type"]


def test_case_variants_match_like_validate_resolves_them(schema):
    # validate folds "transcription profiling"/"Transcription Profiling" onto one spelling
    clean, _ = schema.validate({"assay_measurement": ["transcription profiling"]})
    for acc in ("OSD-1", "OSD-2"):
        matched, _ = schema.dataset_matches(acc, clean)
        assert matched == ["assay_measurement"]


def test_controlled_fields_need_exact_value(schema):
    # word containment is only for anatomy-like fields
    _, contradicted = schema.dataset_matches("OSD-6", {"spaceflight": ["Space Flight"]})
    assert contradicted == ["spaceflight"]
    clean, dropped = schema.validate({"assay_technology": ["microarray"]})
    assert clean == {} and dropped == ["assay_technology=microarray"]


def test_sparse_field_absence_is_not_a_contradiction(schema):
    matched, contradicted = schema.dataset_matches("OSD-1", {"spaceflight": ["Space Flight"]})
    assert matched == [] and contradicted == []


def test_sparse_fields_are_computed_from_the_data(schema):
    # tissue: no dataset has it; spaceflight: 5 of 7 lack it; material_type: only OSD-7 lacks it (1/7)
    assert {"tissue", "spaceflight", "mission"} <= schema.sparse_fields
    assert "material_type" not in schema.sparse_fields
    assert "organism" not in schema.sparse_fields
    assert schema.missing_share["material_type"] == pytest.approx(1 / 7)


def test_missing_mission_is_not_a_contradiction():
    rows = ROWS + [row("OSD-8", organism="Mus musculus", mission="SpaceX-16")]
    s = Schema(rows=rows)
    assert "mission" in s.sparse_fields
    assert s.dataset_matches("OSD-1", {"mission": ["SpaceX-16"]}) == ([], [])
    assert s.dataset_matches("OSD-8", {"mission": ["SpaceX-16"]}) == (["mission"], [])


def test_missing_dense_field_still_contradicts(schema):
    # OSD-7 has no material type; material_type is dense, so asking for liver counts against it
    assert schema.dataset_matches("OSD-7", {"material_type": ["Liver"]}) == ([], ["material_type"])


def test_dataset_without_samples_has_no_evidence(schema):
    assert schema.dataset_matches("OSD-999", {"organism": ["Mus musculus"]}) == ([], [])
