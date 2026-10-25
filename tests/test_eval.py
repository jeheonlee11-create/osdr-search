"""Concept-set labelling and the evaluation's sibling credit, on tiny fixtures."""

from conftest import dataset, make_engine, sample_row

from eval.build_concept_set import build, matches
from eval.run_eval import evaluate_query

MOUSE_LIVER = dataset("OSD-1", "liver study", organism=["Mus musculus"], material_type=["Left Lobe of the Liver"],
                      factors=["Spaceflight"], mission="SpaceX-16; SpaceX-18")
RAT_SOLEUS = dataset("OSD-2", "soleus study", organism=["Rattus norvegicus"], material_type=["Soleus"],
                     factors=["Hindlimb Unloading"], mission="")


def test_criteria_are_anded_and_values_ored():
    assert matches(MOUSE_LIVER, {"organism": ["mus musculus", "Homo sapiens"], "material": ["liver"]})
    assert not matches(MOUSE_LIVER, {"organism": ["Mus musculus"], "factor": ["Hindlimb Unloading"]})
    assert matches(MOUSE_LIVER, {"mission": ["SpaceX-16"]})
    assert not matches(MOUSE_LIVER, {"mission": ["SpaceX-1"]})  # whole tokens only
    assert matches(RAT_SOLEUS, {"text": [r"\bsoleus\b"]})


def test_build_labels_every_matching_dataset():
    built = build([{"id": "C", "query": "q", "criteria": {"organism": ["Mus musculus"]}}],
                  [MOUSE_LIVER, RAT_SOLEUS, dataset("OSD-3", organism=["Mus musculus"])])
    assert built[0]["expected"] == ["OSD-1", "OSD-3"]


TITLE = "Seedling gene network response to DNA damage from radiation in Arabidopsis"
DESC = "Arabidopsis seedlings were irradiated and profiled with several assays."


def test_sibling_of_an_expected_dataset_counts_but_other_tissues_do_not():
    datasets = [
        dataset("OSD-10", TITLE, description=DESC, organism=["Arabidopsis thaliana"], material_type=["Seedlings"],
                assay_technology=["ChIP-seq"]),
        dataset("OSD-11", TITLE, description=DESC, organism=["Arabidopsis thaliana"], material_type=["Seedlings"],
                assay_technology=["RNA-Seq"]),
        dataset("OSD-12", TITLE, description=DESC, organism=["Arabidopsis thaliana"], material_type=["Roots"]),
        dataset("OSD-20", "unrelated"),
    ]
    rows = [sample_row(d["accession"], organism="x") for d in datasets]
    engine = make_engine(datasets, rows, {"OSD-10": 0.9, "OSD-20": 0.8, "OSD-11": 0.7, "OSD-12": 0.6})
    # OSD-11 (RNA-seq) is expected; OSD-10 (ChIP-seq, same seedlings) is shown in its place and counts
    r = evaluate_query(engine, "q", ["OSD-11"], "semantic")
    assert r["shown"] == ["OSD-10", "OSD-20"]
    assert r["correct"] == [True, False]
    assert r["rr@10"] == 1.0 and r["recall@5"] == 1.0
    # the roots dataset is in the same family but a different tissue: no credit
    r = evaluate_query(engine, "q", ["OSD-12"], "semantic")
    assert r["correct"] == [False, False] and r["rr@10"] == 0.0
