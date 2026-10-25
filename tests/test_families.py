"""Near-duplicate grouping and collapsing of search results."""

from conftest import dataset, make_engine, sample_row

from osdr_search.families import build_families, collapse, members_by_family, sibling_key

RR1 = "Rodent Research-1 (RR1) NASA Validation Flight: Mouse {} transcriptomic, proteomic, and epigenomic data"
DESC = "Mice were flown for 37 days on the International Space Station and tissues were collected on orbit."

DATASETS = [
    dataset("OSD-1", RR1.format("liver"), description=DESC, organism=["Mus musculus"], material_type=["Liver"]),
    dataset("OSD-2", RR1.format("kidney"), description=DESC, organism=["Mus musculus"], material_type=["Kidney"]),
    dataset("OSD-3", RR1.format("liver"), description=DESC, organism=["Mus musculus"],
            material_type=["Liver", "Not Applicable"], assay_technology=["mass spectrometry"]),
    dataset("OSD-4", "Arabidopsis root growth on the ISS", description="Seedlings were grown in Veggie.",
            organism=["Arabidopsis thaliana"]),
    # same title, unrelated description: not a duplicate
    dataset("OSD-5", RR1.format("liver"), description="A ground study of diet in aged rats.",
            organism=["Rattus norvegicus"]),
]


def test_near_identical_title_and_description_form_one_family():
    fam = build_families(DATASETS)
    assert fam["OSD-1"] == fam["OSD-2"] == fam["OSD-3"] == "OSD-1"
    assert fam["OSD-4"] == "OSD-4"
    assert fam["OSD-5"] == "OSD-5"
    assert sorted(members_by_family(fam)["OSD-1"]) == ["OSD-1", "OSD-2", "OSD-3"]


def test_sibling_key_separates_tissues_but_ignores_not_applicable():
    fam = build_families(DATASETS)
    by_acc = {d["accession"]: d for d in DATASETS}
    assert sibling_key(by_acc["OSD-1"], fam) == sibling_key(by_acc["OSD-3"], fam)  # liver, different assay
    assert sibling_key(by_acc["OSD-1"], fam) != sibling_key(by_acc["OSD-2"], fam)  # liver vs kidney


def test_collapse_keeps_best_ranked_member_and_folds_the_rest():
    fam = {"a1": "a", "a2": "a", "b": "b", "a3": "a"}
    assert collapse(["a2", "b", "a1", "a3"], fam) == [("a2", ["a1", "a3"]), ("b", [])]


def test_search_output_collapses_families():
    rows = [sample_row(d["accession"], organism=d["organism"][0]) for d in DATASETS]
    engine = make_engine(DATASETS, rows, {"OSD-2": 0.9, "OSD-1": 0.8, "OSD-4": 0.7, "OSD-3": 0.6, "OSD-5": 0.5})
    hits, _, _ = engine.rank("kidney", None, k=3)
    assert [h["accession"] for h in hits] == ["OSD-2", "OSD-4", "OSD-5"]
    assert [r["accession"] for r in hits[0]["related"]] == ["OSD-1", "OSD-3"]
    flat, _, _ = engine.rank("kidney", None, k=3, collapse_families=False)
    assert [h["accession"] for h in flat] == ["OSD-2", "OSD-1", "OSD-4"]
    assert flat[0]["related"] == []
