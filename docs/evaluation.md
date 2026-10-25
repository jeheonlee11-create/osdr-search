# Evaluation

How retrieval quality is measured, what the two test sets are, and what the
numbers do not show. Everything here is reproducible from the repository plus
the gitignored index (`python -m osdr_search.index`); no API key is needed.

```bash
python -m eval.build_concept_set   # rebuild eval/concept.json from the criteria (deterministic)
python -m eval.run_eval            # 2 sets x 3 retrievers -> eval/results_*.json, eval/results_summary.md
```

## Test sets

### Known-item set (`eval/known_item.json`, 50 queries)

The original test set, written during the first build. Most queries paraphrase
one dataset's title or description ("kidney transcriptome from mice flown on the
Rodent Research Reference Mission 2" -> OSD-771), and 47 of 50 have exactly one
expected dataset. It answers one question: *can the system find the dataset I
already have in mind?* It is easy for keyword search, because the query reuses
the title's words, and it says little about open questions where many datasets
are correct. It is kept unchanged and labelled for what it is.

### Concept set (`eval/concept.json`, 32 queries)

Written to test the harder case: a question about a scientific concept, with
every dataset that answers it labelled as correct.

How it was built:

1. **Queries were written by an AI assistant (Claude Code), not by OSDR users or
   curators**, on 2026-10-24, from the distribution of metadata values in
   `data/datasets.json` (organisms, material types, study factors, assays,
   missions). Each targets a concept rather than a title: an organism plus a
   tissue, an assay, a condition, or a mission ("gut bacteria in mouse
   droppings", "rats with their hind legs suspended to mimic weightlessness").
   Queries use everyday words where the metadata uses technical ones (droppings
   vs feces, hind legs suspended vs hindlimb unloading) so they are not a copy of
   the indexed text.
2. Each query carries machine-checkable **criteria** in
   `eval/concept_queries.json`, e.g. C01 "How does going to space change gene
   activity in mouse livers?" requires organism = *Mus musculus*, a material type
   containing the phrase "liver", study factor = Spaceflight, and assay
   measurement = transcription profiling.
3. `eval/build_concept_set.py` labels **every** dataset in the tracked
   `data/datasets.json` that meets all criteria as expected. Nothing is
   hand-picked, and the labels are complete with respect to the metadata.
   Result: 4 to 38 expected datasets per query (median 10, 408 labels in all).
4. Three criteria were changed after looking at label *counts* (before any
   retrieval run): C02 and C23 were widened because they matched only 1-2
   datasets, and C19 switched from organism = *Mus musculus* to a text match on
   "mouse/mice/murine", because mouse fecal metagenomes list their organism as
   "Microbiota". No query or criterion was changed after seeing retrieval results.

Limits of this set:

- Labels are only as good as OSDR's metadata. Datasets with missing or unusual
  values are not labelled (for example mission strings vary: "Bion-M1",
  "BION_M1"), and a dataset that meets the criteria is not necessarily what a
  person asking would want.
- No human relevance judgements were made; the criteria encode one reading of
  each question.
- The queries were written by the same assistant that changed the retrieval code
  and with knowledge of the metadata vocabulary, which can favour keyword
  matching. It is a check, not a benchmark.
- 32 queries is small: one query moves a mean metric by up to 0.03, so
  differences of that size are noise.

## Scoring

Scoring uses the list a user sees: the top 10 results after near-duplicate
families are collapsed (`osdr_search/families.py`).

**Near-duplicates: dedupe, with credit for interchangeable siblings.** OSDR
often splits one study into several datasets with near-identical titles and
descriptions (one per tissue or assay). Two options were considered: count any
family member as a hit, or dedupe the results. Counting any member would credit
the RR-1 kidney dataset (OSD-102) for a question about the RR-1 adrenal gland
dataset (OSD-98), so instead:

- results are deduplicated: each family appears once, as its best-ranked member,
  with the rest listed as "+N related" (this is also what the UI shows);
- a shown result counts as correct when it or a **sibling** is expected, where
  siblings are family members with the same organism and the same material type,
  i.e. they differ only in assay or processing run (the SOG1 ChIP-seq and RNA-seq
  datasets on the same seedlings, or the two plasma-and-cortex mass spectrometry
  datasets OSD-833/OSD-834);
- the best achievable number of correct results is the number of distinct
  families among the expected datasets, since a family only shows once.

Metrics, averaged over queries (`eval/metrics.py`):

- **recall@5** (capped): correct results in the top 5 divided by min(5, number
  of correct families), as BEIR's R_cap@k. Plain recall@5 could not exceed 5/38
  on a query with 38 correct datasets.
- **MRR@10**: 1 / rank of the first correct result, 0 if none is in the top 10.
- **nDCG@10**: binary-relevance discounted cumulative gain, normalised by the best
  possible ordering; rewards correct results more the higher they appear.
- **hit@1 / hit@5**: number of queries with an expected dataset at rank 1 / in
  the top 5 of the *uncollapsed* list with the original labels, the metric the
  project used before, kept for comparison.

## Results (2026-10-24, 641 datasets)

| test set | retriever | queries | recall@5 | MRR@10 | nDCG@10 | hit@1 | hit@5 |
|---|---|---|---|---|---|---|---|
| known_item | semantic | 50 | 0.932 | 0.870 | 0.887 | 40/50 | 48/50 |
| known_item | bm25 | 50 | 0.944 | 0.911 | 0.923 | 44/50 | 48/50 |
| known_item | hybrid | 50 | 0.928 | 0.920 | 0.912 | 45/50 | 48/50 |
| concept | semantic | 32 | 0.480 | 0.656 | 0.484 | 17/32 | 26/32 |
| concept | bm25 | 32 | 0.455 | 0.688 | 0.450 | 17/32 | 29/32 |
| concept | hybrid | 32 | 0.489 | 0.760 | 0.526 | 21/32 | 27/32 |

Reading it:

- On known-item lookup all three retrievers are close; BM25 is as good as or
  better than embeddings because the queries reuse the titles' words. Hybrid has
  the best MRR, and the semantic-only default it replaced had the lowest.
- On concept queries every retriever is far weaker (nDCG@10 about 0.45-0.53),
  which the known-item set hid. Hybrid is best on recall@5, MRR and nDCG@10,
  but the margins over the better single retriever are 0.01-0.07, within or
  close to the noise of 32 queries. BM25 alone puts an expected dataset in the
  uncollapsed top 5 for the most queries (29/32).
- Worst concept queries for hybrid: "mouse studies launched on the SpaceX CRS-16
  resupply flight" (0 of 10 correct: the mission is one token pair among many
  mouse spaceflight studies) and "plant roots grown in space or under altered
  gravity". Both are cases for the structured mission / material filters.

## Not evaluated

- **The LLM step** (filters + rewritten query). It needs `ANTHROPIC_API_KEY`;
  `python -m eval.run_eval --llm` runs it and writes `results_*_llm.json`. The
  filter vote weight (one ranker, weight 1.0) and the rewrite weight (0.5) are
  reasoned defaults, not tuned values.
- **Family thresholds** (title and description Jaccard >= 0.8) were chosen by
  reading the resulting groups, not tuned on either test set.
