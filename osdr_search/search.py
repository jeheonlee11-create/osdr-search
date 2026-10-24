"""The end-to-end pipeline: question -> datasets -> sample-level rows."""

import json
import time

from .fetch import DATASETS_PATH
from .index import VectorIndex
from .llm import QueryTranslator
from .osdr_api import build_url, run
from .schema import Schema

CANDIDATES = 50      # semantic candidates considered before re-ranking
MATCH_BONUS = 0.08   # per validated filter field a dataset satisfies
MISS_PENALTY = 0.12  # per validated filter field a dataset contradicts


class SearchEngine:
    def __init__(self, use_llm=True):
        self.datasets = {d["accession"]: d for d in json.loads(DATASETS_PATH.read_text())}
        self.index = VectorIndex()
        self.schema = Schema()
        self.translator = QueryTranslator(self.schema) if use_llm else None
        self.llm_enabled = bool(self.translator and self.translator.enabled)

    def rank(self, question, structured=None, k=10):
        """Semantic retrieval, then re-rank by validated structured filters."""
        clean, dropped = self.schema.validate(structured.filters() if structured else {})
        clean = {f: v for f, v in clean.items() if f in self.schema.vocab}
        text = structured.search_text if structured and structured.search_text else question
        semantic = self.index.search(text, k=CANDIDATES)
        ranked = []
        for acc, score in semantic:
            matched, contradicted = self.schema.dataset_matches(acc, clean)
            final = score + MATCH_BONUS * len(matched) - MISS_PENALTY * len(contradicted)
            ranked.append((final, score, acc, matched))
        ranked.sort(key=lambda r: -r[0])
        hits = []
        for final, score, acc, matched in ranked[:k]:
            d = self.datasets[acc]
            hits.append({
                "accession": acc,
                "title": d["title"],
                "organism": d["organism"],
                "material_type": d["material_type"],
                "assays": sorted(set(d["assay_technology"])),
                "mission": d["mission"],
                "project_type": d["project_type"],
                "score": round(final, 4),
                "semantic_score": round(score, 4),
                "matched_filters": matched,
                "url": f"https://osdr.nasa.gov/bio/repo/data/studies/{acc}",
            })
        return hits, clean, dropped

    def search(self, question, k=10, fetch_samples=True):
        t0 = time.time()
        structured = None
        llm_error = None
        if self.translator and self.translator.enabled:
            try:
                structured = self.translator.translate(question)
            except Exception as e:  # noqa: BLE001 - degrade to semantic-only
                llm_error = f"{type(e).__name__}: {e}"
        hits, clean, dropped = self.rank(question, structured, k=k)
        result = {
            "question": question,
            "structured_query": structured.model_dump() if structured else None,
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
            except Exception as e:  # noqa: BLE001
                result["samples"] = {"error": f"{type(e).__name__}: {e}", "total": 0, "rows": []}
        result["elapsed_s"] = round(time.time() - t0, 2)
        return result
