# Open Government Data benchmark — start here

Everything below is generated from the data by `scripts/build_docs.py`.
Counts are computed at build time; if one looks stale, re-run the script.

## The generated questions

| what | how many | page | raw data |
|---|---:|---|---|
| **Eurostat questions** (constructed from the article that cites the table) | 1,646 | [questions_eurostat.md](questions_eurostat.md) | [`constructed_se.jsonl`](https://github.com/avishekanand/open-govt-data/blob/benchmark/data/processed/benchmark/constructed_se.jsonl) |
| **CBS questions with executed answers** | 40 | [questions_cbs.md](questions_cbs.md) | [`items_*.jsonl`](https://github.com/avishekanand/open-govt-data/blob/benchmark/data/processed/benchmark/) |
| CBS microdata questions (deferred to CBS) | 1 | [benchmark_items_review.md](benchmark_items_review.md) | — |

## The two benchmarks are separate

CBS and Eurostat have their own questions, their own catalogue and their own
gold answers. They are never mixed: a Dutch question is never answered by a
Eurostat table. Mixing them once put Eurostat datasets in 10.2% of candidate
slots for Dutch questions.

| | CBS StatLine | Eurostat |
|---|---|---|
| question source | research publications that cite CBS data | Statistics Explained articles |
| provenance | resolved from prose citations (lossy) | dataset codes cited verbatim (exact) |
| catalogue | 4,870 tables | 7,438 datasets |
| questions | 40 with executed answers | 1,646 constructed |

## Supporting documents

| document | what it is |
|---|---|
| [methodology.md](methodology.md) | how every artefact was produced, with the defects found on the way |
| [benchmark_design.md](benchmark_design.md) | related work and the multi-hop construction plan |
| [question_analysis.md](question_analysis.md) | what kinds of answers the attested questions admit |
| [benchmark_items_review.md](benchmark_items_review.md) | items for human review, including what was rejected |
| [research_question_examples.md](../data/processed/pub/research_question_examples.md) | the attested questions with their witness sentences |

## The metadata layer these rest on

- **enriched catalogue** — 12,308 records — `data/processed/enriched_unified_qwen3-32b.jsonl` — 12,308 datasets with English titles, descriptions, topics, dimensions
- **field profiles (CBS)** — 4,870 records — `data/processed/field_profiles_cbs.jsonl` — exact period/geography/cardinality per table
- **field profiles (Eurostat)** — 7,438 records — `data/processed/field_profiles_estat.jsonl` — same, for Eurostat
- **surface forms** — 2,108 records — `data/processed/surface_forms_estat.jsonl` — formal / plain / conversational / idiomatic / action-oriented phrasings per table
- **Eurostat article corpus** — 1,309 records — `data/processed/estat/se_articles.jsonl` — Statistics Explained articles and the dataset codes they cite

*Regenerate every page:* `python scripts/build_docs.py`
