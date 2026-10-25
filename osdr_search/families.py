"""Group near-duplicate datasets into families, so one study does not fill the results.

OSDR often splits one study into several OSD entries (one per tissue, assay or
processing run) that share a title and description almost word for word, e.g.
the RR-1 validation flight: adrenal gland, eye, kidney, quadriceps, soleus...
Two datasets are joined when both their title and their description have a
word-set Jaccard similarity of at least 0.8, and families are the connected
components of that relation (on the 641-dataset corpus: 34 families, 101 datasets).

Search output shows the best-ranked member of each family and lists the others
as "related". Within a family, a *sibling group* is the members that also share
organism and tissue, i.e. differ only in assay or processing; evaluation treats
members of one sibling group as interchangeable answers (see docs/evaluation.md).
"""

from collections import defaultdict
from itertools import combinations

from .bm25 import tokenize

TITLE_JACCARD = 0.8
DESCRIPTION_JACCARD = 0.8
_NO_VALUE = {"", "not applicable"}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def build_families(datasets):
    """datasets: list of dataset dicts -> {accession: family id}. A family id is its first member's accession."""
    words = [(d["accession"], set(tokenize(d["title"])), set(tokenize(d.get("description", "")))) for d in datasets]
    parent = {acc: acc for acc, _, _ in words}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for (a, ta, da), (b, tb, db) in combinations(words, 2):
        if ta and tb and _jaccard(ta, tb) >= TITLE_JACCARD and _jaccard(da, db) >= DESCRIPTION_JACCARD:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[rb] = ra
    members = defaultdict(list)
    for acc, _, _ in words:
        members[root(acc)].append(acc)
    return {acc: group[0] for group in members.values() for acc in group}


def members_by_family(family_of):
    out = defaultdict(list)
    for acc, fam in family_of.items():
        out[fam].append(acc)
    return dict(out)


def sibling_key(d, family_of):
    """Datasets with the same key are the same study, organism and tissue: interchangeable answers."""
    def norm(values):
        return frozenset(v.strip().lower() for v in values if v.strip().lower() not in _NO_VALUE)

    return family_of.get(d["accession"], d["accession"]), norm(d["organism"]), norm(d["material_type"])


def collapse(ranked, family_of):
    """ranked: accessions best first -> [(representative, [other accessions of its family that were ranked])].

    The representative is the best-ranked member; later members of the same family are folded into it.
    """
    out, slot = [], {}
    for acc in ranked:
        fam = family_of.get(acc, acc)
        if fam in slot:
            out[slot[fam]][1].append(acc)
        else:
            slot[fam] = len(out)
            out.append((acc, []))
    return out
