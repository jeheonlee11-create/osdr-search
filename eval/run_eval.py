"""Top-k retrieval accuracy on eval/testset.json.

    python -m eval.run_eval            # semantic ranking only (no API key needed)
    python -m eval.run_eval --llm      # LLM filters + re-ranking (needs ANTHROPIC_API_KEY)

A query counts as a hit when any of its expected accessions appears in the top k.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from osdr_search.search import SearchEngine

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use the LLM structured-query step")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    tests = json.loads((HERE / "testset.json").read_text())
    engine = SearchEngine(use_llm=args.llm)
    if args.llm and not engine.llm_enabled:
        sys.exit("--llm requested but ANTHROPIC_API_KEY is not set")
    mode = "llm" if args.llm else "semantic"

    results = []
    hits = {k: 0 for k in (1, 3, args.k)}
    t0 = time.time()
    for t in tests:
        structured = engine.translator.translate(t["query"]) if args.llm else None
        ranked, clean, dropped = engine.rank(t["query"], structured, k=max(hits))
        accs = [h["accession"] for h in ranked]
        rank = next((i + 1 for i, a in enumerate(accs) if a in t["expected"]), None)
        for k in hits:
            hits[k] += bool(rank and rank <= k)
        results.append({
            "query": t["query"], "expected": t["expected"], "rank": rank, "top": accs[:args.k],
            "filters": clean, "dropped": dropped,
        })
        flag = f"rank {rank}" if rank else "MISS"
        print(f"[{flag:>7}] {t['query'][:70]:<70} -> {accs[:3]}")

    n = len(tests)
    print(f"\nmode={mode}  n={n}  " + "  ".join(f"top-{k}: {hits[k]}/{n} = {hits[k]/n:.0%}" for k in sorted(hits)))
    print(f"elapsed {time.time() - t0:.1f}s")
    out = HERE / f"results_{mode}.json"
    out.write_text(json.dumps({"mode": mode, "n": n, "hits": hits, "results": results}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
