"""Build eval/concept.json: concept queries with every dataset that satisfies their metadata criteria.

    python -m eval.build_concept_set

Input is eval/concept_queries.json: a question plus `criteria` describing the
datasets that answer it. Expected answers are not hand-picked: they are every
dataset in data/datasets.json (tracked in git) that meets all criteria, so the
labels are reproducible and complete with respect to the metadata.

Criteria (all listed keys must hold; within a key any value may match):
  organism, factor, assay_technology, assay_measurement
                    case-insensitive exact match on one of the dataset's values
  material          phrase match (whole words, in order) inside a material type,
                    as schema.phrase_in: "liver" matches "Left Lobe of the Liver"
  mission, flight_program, project_type
                    phrase match inside the dataset's string
  text              case-insensitive regex over title + description
"""

import json
import re
from pathlib import Path

from osdr_search.fetch import DATASETS_PATH
from osdr_search.schema import phrase_in

HERE = Path(__file__).resolve().parent
EXACT = ("organism", "factor", "assay_technology", "assay_measurement")
FIELD = {"factor": "factors"}


def _values(d, key):
    v = d[FIELD.get(key, key)]
    return v if isinstance(v, list) else [v]


def matches(d, criteria):
    for key, wanted in criteria.items():
        if key in EXACT:
            have = {v.lower() for v in _values(d, key)}
            ok = any(w.lower() in have for w in wanted)
        elif key == "material" or key in ("mission", "flight_program", "project_type"):
            ok = any(phrase_in(w, v) for w in wanted for v in _values(d, "material_type" if key == "material" else key))
        elif key == "text":
            blob = f"{d['title']} {d['description']}"
            ok = any(re.search(w, blob, re.IGNORECASE) for w in wanted)
        else:
            raise ValueError(f"unknown criterion {key!r}")
        if not ok:
            return False
    return True


def build(queries, datasets):
    out = []
    for q in queries:
        expected = [d["accession"] for d in datasets if matches(d, q["criteria"])]
        out.append({**q, "expected": expected})
    return out


def main():
    queries = json.loads((HERE / "concept_queries.json").read_text())
    datasets = json.loads(DATASETS_PATH.read_text())
    built = build(queries, datasets)
    (HERE / "concept.json").write_text(json.dumps(built, indent=1) + "\n")
    for q in built:
        print(f"{q['id']}  {len(q['expected']):>3} expected  {q['query']}")
    print(f"wrote {len(built)} queries to {HERE / 'concept.json'}")


if __name__ == "__main__":
    main()
