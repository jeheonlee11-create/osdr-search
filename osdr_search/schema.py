"""The metadata schema the LLM output is validated against.

Vocabulary comes straight from the cached sample table, so a filter value is
accepted only if at least one real OSDR sample carries it. Anything else is
dropped (and reported) rather than sent to the API.
"""

import difflib
import json
import re
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

# A field is "sparse" when more than this share of datasets carry no value for
# it at all. On the 641-dataset corpus that is tissue (98% missing), mission
# (51%) and spaceflight (46%); material_type (14%), project_type (<1%),
# organism and the assay fields (0%) are dense. For a sparse field, a dataset
# that lacks the field is "unknown", not a contradiction: half the datasets
# have no mission recorded, which says nothing about which mission they flew.
SPARSE_THRESHOLD = 0.25

# Anatomy-like fields whose values are free-text descriptions of a site
# ("Liver", "Left Lobe of the Liver", "Right soleus"). A filter value matches
# one of these when its words appear, in order and as whole words, inside the
# dataset's value. Every other field (organism, assays, missions, flight
# condition) is a controlled term and only matches case-insensitively exactly:
# "microarray" must not match "protein microarray", "flight" not "pre-flight".
PHRASE_FIELDS = ("material_type", "tissue")


def _tokens(value):
    return tuple(re.findall(r"[a-z0-9]+", str(value).lower()))


def phrase_in(needle, haystack):
    """True when needle's words occur contiguously, as whole words, in haystack.

    "Liver" is in "Left Lobe of the Liver"; it is not in "Delivery", and
    "Lung" is not in "Lunge".
    """
    n, h = _tokens(needle), _tokens(haystack)
    if not n:
        return False
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def value_matches(field, wanted, have):
    """Does a validated filter value match one value a dataset carries?"""
    if field in PHRASE_FIELDS:
        return phrase_in(wanted, have)
    # validate() folds case variants ("Blood"/"blood") onto one spelling, so
    # compare case-insensitively or the other spelling would look contradictory
    return str(wanted).lower() == str(have).lower()


class Schema:
    def __init__(self, rows=None):
        """rows: sample-table records (as in data/samples.json); loaded from disk when omitted."""
        if rows is None:
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
        n = len(self.per_dataset)
        # share of datasets (with any samples) that have no value for each field
        self.missing_share = {f: (sum(f not in fs for fs in self.per_dataset.values()) / n if n else 1.0)
                              for f in QUERY_FIELDS}
        self.sparse_fields = {f for f, share in self.missing_share.items() if share > SPARSE_THRESHOLD}

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
        have = self.per_dataset.get(accession)
        matched, contradicted = [], []
        if have is None:
            return matched, contradicted  # no samples in the table: no evidence either way
        for field, vals in clean.items():
            if field in self.sparse_fields and field not in have:
                continue  # sparse field: absence is not evidence against
            if any(value_matches(field, v, h) for v in vals for h in have.get(field, ())):
                matched.append(field)
            else:
                contradicted.append(field)
        return matched, contradicted
