# OSDR plain-English search

Ask a question in plain English and get back the relevant NASA space biology
datasets, plus the sample-level rows behind them, from the
[Open Science Data Repository](https://osdr.nasa.gov/) (OSDR / GeneLab).

Finding the right dataset in OSDR normally means knowing its metadata schema:
organism, material type, assay technology, mission, factor values. This tool
hides that.

## How it works

```
question
  │
  ├─► Claude (structured output)  ──► proposed filters ──► validated against the
  │        sees the real vocabulary          │              real OSDR vocabulary
  │                                          │              (invalid values dropped)
  └─► bge-small embedding ──► cosine search ─┴─► re-rank ──► top datasets
                               over 641 datasets                  │
                                                                  ▼
                                      OSDR /query/samples/ ◄── validated filters
                                      (live API)               + top accessions
                                                                  │
                                                                  ▼
                                                        sample-level rows
```

1. **Semantic search.** Every OSDR dataset (641 at time of writing) is turned
   into a passage (title, organism, material, factors, assays, mission,
   description) and embedded with `BAAI/bge-small-en-v1.5` (384-d, runs
   locally via ONNX, no API key). Vectors live in a numpy matrix; a query is
   one matrix-vector product.
2. **Natural language → structured query.** Claude (`claude-opus-5-5`, via
   structured outputs) maps the question onto eight OSDR metadata fields
   (organism, material type, tissue, spaceflight condition, assay technology,
   assay measurement, mission, project type). The prompt contains the exact
   vocabulary for each field (pulled from all 51k samples), and every value the
   model returns is validated against that vocabulary again in
   [`schema.py`](osdr_search/schema.py). Values that don't resolve are dropped
   and shown as "dropped" in the UI, so the model cannot invent a field or value
   that doesn't exist.
3. **Re-rank and fetch.** Semantic candidates are re-ranked by how many
   validated filters each dataset satisfies, then the filters plus the top
   accessions are sent to the live OSDR API (`/biodata/api/v2/query/samples/`)
   and the matching samples are returned by a FastAPI app.

## Run it

```bash
python3.11 -m venv ~/.venvs/osdr-search && source ~/.venvs/osdr-search/bin/activate
pip install -r requirements.txt
python -m osdr_search.fetch            # dataset metadata  -> data/datasets.json
python -m osdr_search.fetch samples    # sample table      -> data/samples.json (~30 MB)
python -m osdr_search.index            # embeddings        -> data/index.npz
cp .env.example .env                   # add ANTHROPIC_API_KEY for the LLM step
uvicorn osdr_search.app:app --port 8765
```

Open http://localhost:8765. Without an API key the app still runs, in
semantic-only mode.

API: `GET /api/search?q=...&k=10` returns the structured query, validated and
dropped filters, ranked datasets, and sample rows (capped at 500).

## Evaluation

`eval/testset.json` has 50 natural-language queries, each labelled with the
accession(s) that answer it. A query is a hit if any expected accession is in
the top k.

```bash
python -m eval.run_eval          # semantic only
python -m eval.run_eval --llm    # with LLM filters + re-ranking
```

Results are written to `eval/results_<mode>.json`.

Unit tests (no network, API key or `data/` files needed):

```bash
pip install -r requirements-dev.txt
pytest
```

| mode | top-1 | top-3 | top-5 |
|---|---|---|---|
| semantic only (2026-10-20) | 40/50 = 80% | 48/50 = 96% | 48/50 = 96% |
| semantic + LLM filters | not yet run (needs `ANTHROPIC_API_KEY`) | | |

The two semantic misses are queries that hinge on a mission code ("RR-6")
or an age contrast ("young and old") that the embedding model under-weights;
those are the cases the structured filter step is meant to catch.

Caveats worth saying out loud: the test set is small, the queries were
paraphrased from dataset titles and descriptions (so they're easier than
real user questions), and the labels were written by the author of the test
set, not by OSDR curators. Expanding it with real user questions is the
obvious next step.

## Layout

```
osdr_search/
  fetch.py      download + cache dataset metadata and the sample table
  index.py      embed datasets, cosine search (numpy)
  schema.py     vocabulary + validation of proposed filters
  llm.py        Claude structured-output translation
  osdr_api.py   build + run the live sample query
  search.py     pipeline: translate -> retrieve -> re-rank -> fetch
  app.py        FastAPI app + static/index.html
eval/
  testset.json  50 labelled queries
  run_eval.py   top-k accuracy
tests/          pytest: filter validation + matching on a tiny fixture
```
