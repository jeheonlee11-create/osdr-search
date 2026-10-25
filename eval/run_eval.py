"""Evaluate the retrievers on both test sets and write a results table.

    python -m eval.run_eval                        # all sets x semantic, bm25, hybrid (no API key needed)
    python -m eval.run_eval --sets concept --retrievers hybrid
    python -m eval.run_eval --llm                  # adds LLM filters + rewrite (needs ANTHROPIC_API_KEY)

Test sets (see docs/evaluation.md):
  known_item  50 queries, mostly paraphrases of one dataset's title: "can it find the dataset I mean?"
  concept     32 queries written from metadata concepts, labelled with every dataset meeting the
              query's metadata criteria (eval/build_concept_set.py)

Scoring is on the list a user sees: top 10 results after near-duplicate families are collapsed.
A shown result is correct when its sibling group (same family, organism and tissue; see
osdr_search/families.py) contains an expected dataset. The best possible number of correct results
is the number of distinct families among the expected datasets, since each family shows once.

Writes eval/results_<set>_<retriever>.json (per query) and eval/results_summary.{json,md}.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from osdr_search.families import sibling_key
from osdr_search.search import RETRIEVERS, SearchEngine

HERE = Path(__file__).resolve().parent
SETS = {"known_item": "known_item.json", "concept": "concept.json"}
K = 10


def evaluate_query(engine, query, expected, retriever, structured=None):
    key = {a: sibling_key(d, engine.family_of) for a, d in engine.datasets.items()}
    relevant = {key[a] for a in expected if a in key}
    n_relevant = len({u[0] for u in relevant})  # one slot per family
    hits, clean, dropped = engine.rank(query, structured, k=K, retriever=retriever)
    shown = [key[h["accession"]] for h in hits]
    flat, _, _ = engine.rank(query, structured, k=K, retriever=retriever, collapse_families=False)
    flat_accs = [h["accession"] for h in flat]
    first = next((i for i, a in enumerate(flat_accs, start=1) if a in expected), None)
    return {
        "query": query,
        "expected": expected,
        "shown": [h["accession"] for h in hits],
        "correct": [u in relevant for u in shown],
        "recall@5": recall_at_k(shown, relevant, 5, n_relevant),
        "rr@10": reciprocal_rank(shown, relevant, K),
        "ndcg@10": ndcg_at_k(shown, relevant, K, n_relevant),
        "uncollapsed_first_hit": first,
        "filters": clean,
        "dropped": dropped,
    }


def summarise(rows):
    n = len(rows)
    mean = lambda f: round(sum(r[f] for r in rows) / n, 3)  # noqa: E731
    hit = lambda k: sum(1 for r in rows if r["uncollapsed_first_hit"] and r["uncollapsed_first_hit"] <= k)  # noqa: E731
    return {"n": n, "recall@5": mean("recall@5"), "mrr@10": mean("rr@10"), "ndcg@10": mean("ndcg@10"),
            "hit@1": hit(1), "hit@5": hit(5)}


def markdown(summary):
    lines = ["| test set | retriever | queries | recall@5 | MRR@10 | nDCG@10 | hit@1 | hit@5 |",
             "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['set']} | {s['retriever']} | {s['n']} | {s['recall@5']:.3f} | {s['mrr@10']:.3f} | "
                     f"{s['ndcg@10']:.3f} | {s['hit@1']}/{s['n']} | {s['hit@5']}/{s['n']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", choices=list(SETS), default=list(SETS))
    ap.add_argument("--retrievers", nargs="+", choices=RETRIEVERS, default=["semantic", "bm25", "hybrid"])
    ap.add_argument("--llm", action="store_true", help="also use the LLM structured-query step")
    args = ap.parse_args()

    engine = SearchEngine(use_llm=args.llm)
    if args.llm and not engine.llm_enabled:
        sys.exit("--llm requested but ANTHROPIC_API_KEY is not set")
    suffix = "+llm" if args.llm else ""

    summary = []
    t0 = time.time()
    for set_name in args.sets:
        tests = json.loads((HERE / SETS[set_name]).read_text())
        # translate once per query so every retriever sees the same LLM output
        structured = [engine.translator.translate(t["query"]) if args.llm else None for t in tests]
        for retriever in args.retrievers:
            rows = [evaluate_query(engine, t["query"], t["expected"], retriever, sq)
                    for t, sq in zip(tests, structured, strict=True)]
            s = {"set": set_name, "retriever": retriever + suffix, **summarise(rows)}
            summary.append(s)
            out = HERE / f"results_{set_name}_{retriever}{suffix.replace('+', '_')}.json"
            out.write_text(json.dumps({**s, "results": rows}, indent=1) + "\n")
            print(f"{set_name:<10} {retriever + suffix:<12} recall@5 {s['recall@5']:.3f}  MRR {s['mrr@10']:.3f}  "
                  f"nDCG@10 {s['ndcg@10']:.3f}  -> {out.name}")

    table = markdown(summary)
    name = f"results_summary{suffix.replace('+', '_')}"
    (HERE / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n")
    (HERE / f"{name}.md").write_text(table + "\n")
    print(f"\n{table}\n\nelapsed {time.time() - t0:.1f}s; wrote eval/{name}.md")


if __name__ == "__main__":
    main()
