"""The end-to-end pipeline: question -> datasets -> sample-level rows."""

import json
import logging
import time

from .bm25 import BM25
from .families import build_families, collapse, members_by_family
from .fetch import DATASETS_PATH
from .fusion import RRF_K, rrf
from .index import VectorIndex, check_fingerprint, doc_text
from .llm import QueryTranslator, accept_search_text
from .osdr_api import SampleQueryTooLarge, build_url, run
from .schema import Schema

log = logging.getLogger(__name__)

RETRIEVERS = ("hybrid", "semantic", "bm25")
CANDIDATES = 50  # candidates taken from each ranker before fusion and re-ranking
# The validated LLM filters vote in the same rank fusion as BM25 and the
# embeddings, as one more ranker with this weight (see SearchEngine.rerank).
FILTER_WEIGHT = 1.0
# Rankings of the LLM's rewritten search_text get half a vote each, so the rewrite
# can refine the ranking but never outvote the user's own words.
REWRITE_WEIGHT = 0.5
# What the browser sees when a step fails; the exception itself is only logged server-side.
LLM_ERROR = "The LLM step failed, so these results use keyword + semantic ranking without filters."
SAMPLES_TOO_LARGE = "Too many matching samples to show here; ask for fewer datasets or a narrower question."
SAMPLES_ERROR = "The live OSDR sample query failed or timed out. The samples URL above can be opened directly."


def _study_url(acc):
    return f"https://osdr.nasa.gov/bio/repo/data/studies/{acc}"


def _accession_number(acc):
    return int(acc.split("-")[1]) if acc.split("-")[-1].isdigit() else 0


class SearchEngine:
    def __init__(self, use_llm=True, retriever="hybrid", *, datasets=None, index=None, schema=None,
                 translator=None):
        """Loads data/ files for any component not passed in (tests pass small fakes)."""
        if retriever not in RETRIEVERS:
            raise ValueError(f"retriever must be one of {RETRIEVERS}")
        self.retriever = retriever
        if datasets is None:
            datasets = json.loads(DATASETS_PATH.read_text())
        self.datasets = {d["accession"]: d for d in datasets}
        self.bm25 = BM25(self.datasets, [doc_text(d) for d in self.datasets.values()])
        self.family_of = build_families(datasets)
        self.family_members = members_by_family(self.family_of)
        self.index = index if index is not None else VectorIndex()
        if hasattr(self.index, "fingerprint"):
            check_fingerprint(self.index, datasets)  # fail now, with a rebuild hint, not with a KeyError later
        self.schema = schema if schema is not None else Schema()
        no_samples = [a for a in self.datasets if a not in self.schema.per_dataset]
        if no_samples:
            log.warning("%d datasets have no rows in the sample table (e.g. %s); filters cannot match them. "
                        "Refresh it with `python -m osdr_search.fetch samples`.", len(no_samples), no_samples[:3])
        if translator is None and use_llm:
            translator = QueryTranslator(self.schema)
        self.translator = translator
        self.llm_enabled = bool(self.translator and self.translator.enabled)

    def warmup(self):
        """Load the embedding model now rather than on the first user's query."""
        self.index.search("warmup", k=1)

    def rankings(self, text, retriever=None):
        """{ranker name: [accession, ...]} for one piece of query text."""
        retriever = retriever or self.retriever
        out = {}
        if retriever in ("hybrid", "semantic"):
            out["semantic"] = [a for a, _ in self.index.search(text, k=CANDIDATES)]
        if retriever in ("hybrid", "bm25"):
            out["bm25"] = [a for a, _ in self.bm25.search(text, k=CANDIDATES)]
        return out

    def rerank(self, fused, clean):
        """Add the validated filters as one more ranker in the fusion.

        Candidates are ordered by (filters matched - filters contradicted), ties
        kept in fused order, and that list gets an RRF vote like any other
        ranker; exact ties are then broken by the filter evidence. A
        rank-based vote is on the same scale as the fused scores whatever the
        retrievers' raw scores look like, so there is no hand-set bonus to
        recalibrate when the retriever changes.
        """
        if not clean:
            return [(acc, score, []) for acc, score in fused]
        evidence = {acc: self.schema.dataset_matches(acc, clean) for acc, _ in fused}
        net = {acc: len(m) - len(c) for acc, (m, c) in evidence.items()}
        order = {acc: i for i, (acc, _) in enumerate(fused)}
        by_filters = sorted(order, key=lambda a: (-net[a], order[a]))
        bonus = {acc: FILTER_WEIGHT / (RRF_K + r) for r, acc in enumerate(by_filters, start=1)}
        out = [(acc, score + bonus[acc], evidence[acc][0]) for acc, score in fused]
        # exact ties (common with a single retriever) go to the dataset with more filter evidence
        return sorted(out, key=lambda r: (-round(r[1], 12), -net[r[0]], order[r[0]]))

    def rank(self, question, structured=None, k=10, retriever=None, collapse_families=True):
        """Retrieve (BM25 and/or embeddings, fused by RRF), re-rank by validated filters, collapse families.

        With collapse_families, each near-duplicate family (families.py) takes one slot: its best-ranked
        member, with the family's other datasets listed under "related".
        """
        clean, dropped = self.schema.validate(structured.filters() if structured else {})
        clean = {f: v for f, v in clean.items() if f in self.schema.vocab}
        lists = self.rankings(question, retriever)
        # the LLM's rewrite adds votes next to the original question; it never replaces it
        rewrite = accept_search_text(question, structured.search_text) if structured else None
        if rewrite and rewrite.lower() != question.lower():
            lists.update({f"{name}_rewrite": ids for name, ids in self.rankings(rewrite, retriever).items()})
        fused = rrf(lists, weights={name: REWRITE_WEIGHT for name in lists if name.endswith("_rewrite")})
        ranked = self.rerank(fused, clean)
        positions = {name: {a: i for i, a in enumerate(ids, start=1)} for name, ids in lists.items()}
        info = {acc: (score, matched) for acc, score, matched in ranked}
        if collapse_families:
            groups = collapse([acc for acc, _, _ in ranked], self.family_of)[:k]
        else:
            groups = [(acc, []) for acc, _, _ in ranked[:k]]
        hits = []
        for acc, folded in groups:
            score, matched = info[acc]
            d = self.datasets[acc]
            fam = self.family_members.get(self.family_of.get(acc), [acc]) if collapse_families else [acc]
            related = folded + sorted((a for a in fam if a != acc and a not in folded), key=_accession_number)
            hits.append({
                "accession": acc,
                "title": d["title"],
                "organism": d["organism"],
                "material_type": d["material_type"],
                "assays": sorted(set(d["assay_technology"])),
                "mission": d["mission"],
                "project_type": d["project_type"],
                "score": round(score, 5),
                "ranks": {name: pos.get(acc) for name, pos in positions.items()},
                "matched_filters": matched,
                "url": _study_url(acc),
                "related": [{"accession": a, "material_type": self.datasets[a]["material_type"],
                             "assays": sorted(set(self.datasets[a]["assay_technology"])), "url": _study_url(a)}
                            for a in related],
            })
        return hits, clean, dropped

    def search(self, question, k=10, fetch_samples=True):
        t0 = time.time()
        structured = None
        llm_error = None
        if self.translator and self.translator.enabled:
            try:
                structured = self.translator.translate(question)
            except Exception:  # noqa: BLE001 - degrade to retrieval-only
                # details go to the server log only; exception text can carry URLs, keys or internals
                log.exception("LLM translation failed")
                llm_error = LLM_ERROR
        hits, clean, dropped = self.rank(question, structured, k=k)
        result = {
            "question": question,
            "retriever": self.retriever,
            "structured_query": structured.model_dump() if structured else None,
            "search_text_used": bool(structured and accept_search_text(question, structured.search_text)),
            "validated_filters": clean,
            "dropped_filters": dropped,
            "llm_used": structured is not None,
            "llm_error": llm_error,
            "datasets": hits,
            "samples": None,
            "samples_url": None,
        }
        if fetch_samples and hits:
            url = build_url(clean, accessions=[h["accession"] for h in hits])
            result["samples_url"] = url
            try:
                samples = run(url)
                # present rows in the same order as the ranked datasets
                order = {h["accession"]: i for i, h in enumerate(hits)}
                samples["rows"].sort(key=lambda r: order.get(r.get("accession"), len(order)))
                result["samples"] = samples
            except SampleQueryTooLarge:
                log.warning("OSDR sample query too large: %s", url)
                result["samples"] = {"error": SAMPLES_TOO_LARGE, "total": 0, "rows": []}
            except Exception:  # noqa: BLE001
                log.exception("OSDR sample query failed: %s", url)
                result["samples"] = {"error": SAMPLES_ERROR, "total": 0, "rows": []}
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result
