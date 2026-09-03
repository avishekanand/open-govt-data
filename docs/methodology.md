# Methodology: how the question corpus and benchmark were built

A full provenance record, written for oversight. Every count below is measured
from the artefacts in this repository, and every stage names the script that
produced it so any figure can be re-derived or challenged.

**Nothing here has been validated by a human domain expert yet.** All labels are
model-assigned. Section 9 lists what that means for interpretation.

### Review these

| document | what to check |
|---|---|
| **[benchmark_items_review.md](benchmark_items_review.md)** | the cleaned question → dataset → SQL → **answer** pairs, end to end, plus the `gold_ready` candidates awaiting SQL and a spot-check of what was excluded |
| [research_question_examples.md](../data/processed/pub/research_question_examples.md) | the extracted questions themselves, with the verbatim witness sentence and source publication |
| [question_analysis.md](question_analysis.md) | what kinds of answers the 3,276 questions admit, and why only a minority are verifiable |
| [benchmark_design.md](benchmark_design.md) | related work and the intended multi-hop construction procedure |

Benchmark work continues on the **`benchmark`** branch.

---

## 0. Data flow

```
publications workbook  ──▶ pub_ingest    ──▶ 2,217 unique URLs
                                              │
                            pub_download  ◀───┘   crawl, extract full text
                                 │
                                 ├──▶ data/raw/pub_text/*.txt   (evidence store, git-ignored)
                                 └──▶ documents.parquet          1,513 OK
                                          │
                            pub_evidence  ◀┘   chunk → extract → merge → verify → link
                                 │
                                 └──▶ pub_evidence.jsonl   1,176 docs · 3,276 questions · 3,329 mentions
                                          │
             translate_questions ─────────┤   + question_en
             classify_questions ──────────┤   + answer_type, data_needed, verifiable_now
             triage_questions ────────────┘   + benchmark_status, attributed_dataset
                                          │
                            build_seed_sets ──▶ seeds_cbs / seeds_estat / seeds_rejected
                                          │
                                     bench/ ──▶ items_v0.jsonl  (gold SQL + pinned snapshot)
```

The dataset catalogue used for linking (12,308 datasets) is built separately by
`enrich.ingest_eurostat_meta`, `cbs.batch_ingest_statline`, `enrich.ingest_cbs_codes`
and `enrich.run_vllm` — see [README](../README.md).

---

## 1. Source and URL extraction — `cbs.pub_ingest`

Input: `Publications_overview_internet_May_26.xlsx` (not in git; the derived
parquets are).

| | count |
|---|---:|
| publication records | 3,106 |
| unique URLs after normalise + dedup | 2,217 |

URLs come from two columns, tagged `url_kind` = `publication_url` / `code_url`,
and are classified by pattern into `resource_type` (html/pdf/doi/github/zenodo).
Deduplication keeps the list of referencing record ids, so a URL cited by several
publications is crawled once.

## 2. Crawl and text extraction — `cbs.pub_download`

| | count |
|---|---:|
| URLs fetched | 2,217 |
| **extracted OK** | **1,513** |
| failed (404 / timeout / blocked / invalid) | 704 |

By type (OK): html 1,196 · pdf 228 · doi 84 · github 3 · zenodo 2.
Total text **47.6M characters**; median document 3,409, p90 86,034, max 4,315,552.

HTML via BeautifulSoup (`lxml`, falling back to `html.parser`), stripping
`script/style/nav/footer/header/noscript`. PDF via `pypdf`. Full text is written
to `data/raw/pub_text/<sha1(url)>.txt` — git-ignored, re-crawlable.

**Three defects fixed at this stage** (all would have silently truncated the corpus):

1. `lxml` was not installed, so `BeautifulSoup(content, "lxml")` raised
   `FeatureNotFound` — which fails **every HTML document**, i.e. 1,196 of 1,513.
   Now pinned in `requirements.txt`, with a stdlib fallback.
2. `MAX_TEXT = 40000` truncated every document at 40k characters.
3. `reader.pages[:40]` truncated PDFs at 40 pages.

Caps 2 and 3 are now off by default (`PUB_MAX_TEXT`, `PUB_MAX_PDF_PAGES` restore
them). Verified effect: maximum extracted text on a fixed sample rose from exactly
40,000 to 92,501 characters.

## 3. Chunking and relevance filter — `enrich.pub_evidence`

- **Eligibility**: `ok == True` and `text_len >= 300` → **1,176 documents**.
- **Chunking**: 6,000-character windows, 600-character overlap, cut at the nearest
  paragraph/sentence boundary past the halfway point. Capped at 40 chunks per
  document (240k characters) so one 4.3M-character book cannot dominate a batch.
- **Relevance filter**: a chunk is sent to the model only if it matches a
  CBS/Eurostat signal regex (`cbs`, `statline`, `microdata`, a table-id pattern,
  register names, `eurostat`, …) **or** is among the first 2 chunks, where a paper
  states its aim.

Effect: **7,209 chunks → 4,000** (−44%), removing mostly boilerplate.
Override with `--all-chunks`.

## 4. Extraction — `enrich.pub_evidence`

One constrained-JSON generation per chunk, extracting `data_mentions`,
`research_questions`, `uses_cbs_data`, `data_kind`. **Every item must carry a
witness sentence copied verbatim from the chunk**; the prompt instructs the model
to omit an item rather than invent a quote.

Merge rules across a document's chunks:

| rule | rationale |
|---|---|
| mentions deduped on normalised text; verified witness preferred | same table named in several chunks |
| mentions longer than **80 characters** dropped | a dataset reference is a name, not a copied sentence fragment |
| questions taken **only from chunks 1–2**, capped at **8** per document | without this, a 40-chunk book yielded 135 questions, mostly rhetorical asides (72.7/doc corpus-wide) |
| `data_kind` = `both` requires **≥2** supporting chunks per side | one stray mention otherwise flipped 23 of 25 documents to "both" |

Run: 1,116 documents / 3,833 chunks in 145.1 minutes; **91 unparsed chunks (2.4%)**.
Output **1,176 documents · 3,329 mentions · 3,276 questions**.

## 5. Witness verification — `enrich.pub_evidence.verify_witness`

After generation, each witness is checked to occur in the source text
(whitespace-normalised, case-insensitive; a 60% inner span is accepted to tolerate
OCR drift). Failures are **kept and flagged**, never silently dropped.

| | verified |
|---|---:|
| mentions | 90.2% |
| questions | 91.2% |

This proves a sentence exists in the document. It does **not** prove the model
read it correctly.

## 6. Dataset linking — `enrich.pub_link`

Three strategies, most reliable first: **exact code** (`83765NED`,
`nama_10r_3gdp`) → **register** (GBA/BRP, POLIS, … deliberately *not* linked to a
table, since registers have no public table) → **fuzzy title** (Jaccard over
stopword-stripped tokens against native *and* LLM-enriched English titles, so
English prose can reach Dutch titles).

| outcome | count |
|---|---:|
| unresolved | 2,980 |
| register | 287 |
| fuzzy_title | 52 |
| exact_id | 10 |
| **resolved to a table** | **62, across 49 distinct codes, in 37 documents** |

Fuzzy scores: min 0.60, median 0.88; **37 of 52 are ≥ 0.7**. Downstream stages
apply `--min-link-score 0.7`, which discards matches driven by generic citation
strings (`"Statistics Netherlands, 2022"` → `85067NED`).

**Rejected alternative:** containment scoring was tried instead of Jaccard to
catch referring phrases. It was reverted — it scores a short title that is a
subset of the mention at 1.0, so *"GDP by NUTS 3 region"* matched *"Regional
Population for GDP Calculation by NUTS 3 Region"* ahead of the actual GDP table.
Stopword removal, not a softer metric, was the correct fix.

**Known ceiling.** Lexical matching cannot bridge vocabulary: *"weekly deaths"*
never reaches *"Deceased Persons by Gender and Age Group, per Week"*. This is the
main constraint on benchmark size, not question supply.

## 7. Enrichment passes over questions

All three use the same configuration (§8) and are idempotent and atomic
(temp file + `replace`), so a re-run fills gaps rather than rewriting.

| pass | script | adds | result |
|---|---|---|---|
| translation | `enrich.translate_questions` | `question_en` | 3,276/3,276, 0 failures, 1.8 min |
| classification | `enrich.classify_questions` | `answer_type`, `data_needed`, `publisher_hint`, `verifiable_now`, `reason` | 3,276/3,276, 0 failures, 7.8 min |
| triage | `enrich.triage_questions` | `benchmark_status`, `attributed_dataset`, `attribution_confidence`, `specificity`, `missing_to_specify` | see §10 |

**Witness sentences are never translated.** They are evidence verified by exact
match against the source; a translated quote could not be checked.

Classification results (details in [question_analysis.md](question_analysis.md)):
`verifiable_now` **628 / 3,276 = 19.2%**; `data_needed` = microdata 1,202 /
other_source 1,054 / public_aggregate 1,008 / none 12; `publisher_hint` = CBS
1,998 / not_applicable 1,230 / either 41 / **ESTAT 7**.

## 8. Model and decoding configuration

| | |
|---|---|
| model | `Qwen/Qwen3-32B`, bfloat16 |
| runtime | vLLM 0.23.0, 1× A100-80GB (SLURM, `scripts/*.slurm`) |
| structured output | JSON-schema-constrained; schema validated against **both** xgrammar and llguidance before each run |
| thinking mode | `enable_thinking=False` (Qwen3 is hybrid; otherwise it emits a `<think>` block) |
| temperature | extraction 0.1 · translation 0.1 · classification 0.0 · triage 0.0 |
| reproducibility | temperature 0 for label passes; enums prevent value drift |

**Why dual-backend validation exists:** vLLM chooses its structured-output backend
automatically. `uniqueItems` compiles under xgrammar but hard-fails the entire job
under llguidance — discovered only because a smoke test ran on the GPU. Schema
validation now runs in `--dry-run` on CPU in seconds.

## 9. Known defects, and how they were caught

| # | defect | effect | caught by | status |
|---|---|---|---|---|
| 1 | `lxml` missing | every HTML document fails | crawl smoke test | fixed + pinned |
| 2 | 40k char / 40 page caps | silent truncation | inspecting `text_len` max = exactly 40000 | removed |
| 3 | `uniqueItems` unsupported by llguidance | whole job aborts | GPU smoke test | schema fixed; dual validation added |
| 4 | `--limit` sorts by size | "smoke test" ran the 25 largest docs, 38 min | reading the log | added `--sample` |
| 5 | questions from every chunk | 72.7 questions/doc, mostly asides | output inspection | intro-only + cap 8 |
| 6 | `data_kind` "both" over-triggered | 92% of docs mislabelled | output inspection | ≥2 chunks per side |
| 7 | mentions were sentence fragments | noise in linking | output inspection | 80-char cap |
| 8 | **document-level dataset attribution** | every question inherited every dataset the paper cited; 4 kidney-disease questions paired with an income table | attempting to author gold SQL | fixed by `triage_questions` (per-question attribution) |
| 9 | weak fuzzy links kept | generic citation strings linked | manual review of the 14 candidates | `--min-link-score 0.7` |
| 10 | first-*k* category sampling (catalogue side) | queries skewed to alphabetically first values (all-Albanian regions) | reviewing enrichment output | `enrich.sampling.spread_sample` |
| 11 | case mismatch TOC vs vendored CSV | Eurostat yielded 0 items | dry run showed `{'CBS': 4870}` only | filter removed |
| 12 | **triage `benchmark_status` miscalibrated, twice** | v1 put 73% of questions in `not_a_data_question` (incl. 545 the classifier called `public_aggregate`) and never once used `ambiguous`; v2, after the taxonomy was separated, was **worse** at 91.4% and 94.7% `vague` | cross-tabulating the triage labels against the classifier labels | **field abandoned** — see below |

### Defect 12 in detail: a label we decided not to trust

The triage pass was meant to assign a benchmark status per question. Two runs
disagreed with the classifier pass on the same underlying property, in the same
direction, and got worse rather than better after a fix:

| | v1 | v2 (after taxonomy fix) |
|---|---:|---:|
| `not_a_data_question` | 73.0% | **91.4%** |
| `vague` | 74.9% | **94.7%** |
| classifier `public_aggregate` routed to `not_a_data_question` | 545 | **867** |

Two failures of the same design are a signal about the design, not the prompt.
The rubric asked whether a question has an explicit population, period and
measure — a bar real research questions almost never meet, since *"Which
conditions lead to the highest healthcare expenditures?"* is a perfectly ordinary
data question that simply needs a period pinned.

**Decision: the `benchmark_status` and `specificity` fields are not used
anywhere downstream.** Data availability is taken from the classifier pass
(`data_needed`), whose distribution — microdata 1,202 / other_source 1,054 /
public_aggregate 1,008 — is at least facially plausible.

What *did* survive from triage is the part it was built for: **per-question
dataset attribution**. Across 147 questions carrying candidates it attributed 21,
**invented zero codes**, and 8 of those are high-confidence. Requiring three
independent signals to agree — classifier says `public_aggregate`, classifier
says `verifiable_now`, triage attributes a dataset to *this* question — yields
**5 candidates**, listed for review in
[benchmark_items_review.md](benchmark_items_review.md) §3a.

This is the honest replacement for the "14 ready pairs" figure: **5**, each
resting on agreement between two independently-prompted passes, awaiting human
confirmation.

Defect 8 is the reason the "14 ready pairs" figure quoted earlier is wrong; §10
supersedes it.

## 10. Seed sets and benchmark items

`scripts/build_seed_sets.py` splits classified questions into
`data/processed/benchmark/seeds_{cbs,estat,rejected}.jsonl`. **Rejected items keep
`rejected_because`**, so the filter is auditable and revisable.

⚠️ The current seed files were built **before** the triage pass and therefore carry
document-level `candidate_datasets`. They must be regenerated on
`attributed_dataset` once triage completes; counts here will change.

Authored items are rendered for human review by
`scripts/make_benchmark_review.py` into
[benchmark_items_review.md](benchmark_items_review.md) — question, clarified
question, dataset, gold SQL, and the executed answer, so a reviewer can reject
any link in the chain.

`bench/` provides the executable layer:

- `bench/substrate.py` — DuckDB views over statistical tables, materialised on
  demand. Every CBS table shares one canonical long schema
  (`measure, unit, value, status, <Dim>, <Dim>_label`), so one SQL dialect covers
  all 12,308 datasets and multi-hop is a JOIN on shared dimension labels.
- `bench/scoring.py` — **execution match, not query-string match**: one question
  admits many correct queries. Results are canonicalised (lowercased columns,
  numerics rounded to 3 dp, rows sorted) and compared by containment on the gold
  columns. BIRD-strict column order would fail correct answers to underspecified
  questions; Spider 2.0 uses the looser rule and we follow it.

Item tiers:

| tier | meaning |
|---|---|
| `A_executable` | gold SQL runs now against public aggregate tables; result snapshot pinned with retrieval date and table `last_update` |
| `B_microdata_deferred` | gold SQL authored against documented register schemas, `executable: false`, no snapshot — to be validated inside the CBS secure environment |

Tier B exists so the 1,202 microdata questions — the largest single bucket —
remain in the benchmark as a "write the query you would run" task.

## 10b. Contextualisation and embedding retrieval (iteration 2)

Two passes built to attack the 628 -> 21 collapse, where 607 usable questions
were lost not because they were bad but because their publication cited no
dataset we could link.

**`enrich.contextualize_questions`** rewrites each question into a self-contained
form using the document window around its own witness sentence. The verbatim
original is never overwritten. 3,276/3,276, no failures.

| scope field recovered | share |
|---|---:|
| measure | 98% |
| geography | 83% |
| population | 83% |
| period | 47% |

Confidence high 2,371 / medium 737 / low 168. This confirmed that the "94.7%
vague" verdict of defect 12 was largely an artefact of our own extraction:
*"Which conditions lead to the highest healthcare expenditures?"* becomes
*"...in the Netherlands in 2019?"* once its own paragraph is restored.

Two weaknesses to watch: `period` is the least recoverable field (47%) and is
precisely what gold SQL must pin; and one rewrite appended a data-source name
("based on data from the Mulier Instituut"), which would leak provenance into
the question text if it happened at scale.

**`enrich.embed_link`** embeds every question against all 12,308 enriched
datasets (`bge-large-en-v1.5`, CLS pooling, query-instruction prefix; dataset
text = `title_en | title_native | topics | description | dimension names`). This
removes the requirement that the publication cited anything.

The built-in sanity check reported **hit@1 0/9, hit@10 0/9** against the
lexically-attributed pairs. On inspection this is **not** a retrieval failure —
it is a disagreement about what gold means:

| | cited (lexical) | retrieved (embedding) |
|---|---|---|
| liveability pressure | `85067NED` *Regions in the Netherlands* — a geography reference table | `81924NED` *Liveability and Nuisance in Neighborhoods* |
| youth employment after covid | `83031NED` *Labour Participation by Education Level* | `86087NED` *Youth Labour Market Situation (15-27)* |

Publications cite reference tables (region definitions, classifications)
alongside the table that carries the measure. So:

- **provenance-faithful gold** = what the study used → right for *recompute the
  paper's number*
- **task-faithful gold** = what best answers the question → right for a
  *retrieval/discovery* benchmark

`hit@k` against lexical attribution measures agreement between two different
targets and should not be read as retrieval quality. Both candidate sets are put
side by side for human adjudication in
[benchmark_items_review.md](benchmark_items_review.md) §4; **which notion is gold
is a decision, not a computation.**

Known defect in retrieval: *"ICT Usage in Small Businesses by Company Size"*
ranked first for two youth-employment questions, so top-1 auto-accept would be
wrong. Candidates are evidence for review, never gold.

## 11. Threats to validity

- **No human validation.** Every label is model-assigned. Temperature 0 makes them
  reproducible, not correct. A human check on a sample, with agreement reported,
  is required before any of this is called gold.
- **`verifiable_now` is a guess about the world.** The classifier judged whether a
  suitable public table plausibly exists; it never queried the catalogue. A
  retrieval-backed check would be stricter.
- **Witness verification proves existence, not comprehension.**
- **Corpus skew.** Dutch policy research: `publisher_hint` = ESTAT for only 7 of
  3,276. A Eurostat arm needs a different seed source, not a filter over this one.
- **Genre mix.** Parliamentary *Kamerstukken* contribute numbered `Vraag 1…` items
  that are questions to a minister, not research aims. Labelled, not removed.
- **704 unreachable URLs** are not missing at random — dead links skew old, so the
  corpus tilts recent.
- **Answers drift.** Official statistics are revised; one table declared
  `dataend: 2023` while serving data to 2025. Snapshots are pinned for this reason.

## 12. Reproduction

```bash
python -m cbs.pub_ingest                       # workbook -> URLs
python -m cbs.pub_download --workers 10        # crawl + full text
python -m enrich.pub_evidence --dry-run        # sizing + schema validation
sbatch scripts/pub_evidence.slurm              # extraction (GPU)
sbatch scripts/translate_questions.slurm
sbatch scripts/classify_questions.slurm
sbatch scripts/triage_questions.slurm
python scripts/build_seed_sets.py
python scripts/make_question_examples.py
```

Intermediate snapshots taken before each destructive pass are kept outside git;
the in-place passes are atomic and idempotent, so any stage can be re-run.
