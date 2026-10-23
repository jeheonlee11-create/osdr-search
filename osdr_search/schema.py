"""The metadata schema the LLM output is validated against.

Vocabulary comes straight from the cached sample table, so a filter value is
accepted only if at least one real OSDR sample carries it. Anything else is
dropped (and reported) rather than sent to the API.
"""

import difflib
import json
from collections import defaultdict

from .fetch import SAMPLES_PATH

# Friendly field name used in the LLM schema -> canonical OSDR query field.
QUERY_FIELDS = {
    "organism": "study.characteristics.organism",
    "material_type": "study.characteristics.material type",
    "tissue": "study.factor value.tissue",
    "spaceflight": "study.factor value.spaceflight",
    "assay_technology": "investigation.study assays.study assay technology type",
    "assay_measurement": "investigation.study assays.study assay measurement type",
    "mission": "investigation.study.comment.mission name",
    "project_type": "investigation.study.comment.project type",
}

FUZZY_CUTOFF = 0.88


class Schema:
    def __init__(self):
        if not SAMPLES_PATH.exists():
            raise FileNotFoundError(f"{SAMPLES_PATH} missing; run `python -m osdr_search.fetch samples`")
        rows = json.loads(SAMPLES_PATH.read_text())
        self.sample_count = len(rows)
        # vocab[field] -> sorted list of accepted values
        # per_dataset[accession][field] -> set of values seen in that dataset's samples
        values = defaultdict(set)
        per_dataset = defaultdict(lambda: defaultdict(set))
        for r in rows:
            acc = r["id.accession"]
            for field, column in QUERY_FIELDS.items():
                v = r.get(column)
                if v in (None, "", "None"):
                    continue
                v = str(v)
                values[field].add(v)
                per_dataset[acc][field].add(v)
        self.vocab = {f: sorted(values[f], key=str.lower) for f in QUERY_FIELDS}
        self._lower = {f: {v.lower(): v for v in vs} for f, vs in self.vocab.items()}
        self.per_dataset = {a: dict(fs) for a, fs in per_dataset.items()}

    def resolve(self, field, value):
        """Map a proposed value onto a real vocabulary value, or None."""
        if field not in self.vocab or not value:
            return None
        value = str(value).strip()
        if value in self._lower[field].values():
            return value
        if value.lower() in self._lower[field]:
            return self._lower[field][value.lower()]
        close = difflib.get_close_matches(value.lower(), list(self._lower[field]), n=1, cutoff=FUZZY_CUTOFF)
        return self._lower[field][close[0]] if close else None

    def validate(self, proposed):
        """proposed: {field: [values]} -> (clean {field: [values]}, dropped ["field=value", ...])."""
        clean, dropped = {}, []
        for field, vals in (proposed or {}).items():
            if field not in QUERY_FIELDS:
                dropped.append(f"{field}=* (unknown field)")
                continue
            kept = []
            for v in vals or []:
                r = self.resolve(field, v)
                if r is None:
                    dropped.append(f"{field}={v}")
                elif r not in kept:
                    kept.append(r)
            if kept:
                clean[field] = kept
        return clean, dropped

    def dataset_matches(self, accession, clean):
        """Which validated filter fields does this dataset satisfy (and which does it contradict)?"""
        have = self.per_dataset.get(accession, {})
        matched, contradicted = [], []
        for field, vals in clean.items():
            if field in ("spaceflight", "tissue") and field not in have:
                continue  # sparse fields: absence is not evidence against
            if have.get(field, set()) & set(vals):
                matched.append(field)
            else:
                contradicted.append(field)
        return matched, contradicted
