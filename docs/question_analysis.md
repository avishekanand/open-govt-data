# What the 3,276 attested questions actually admit as answers

Analysis of the research questions extracted from 1,176 publications, classified
by `enrich.classify_questions` (Qwen3-32B, temperature 0, schema-constrained).
The point of this pass is to find the subset that can seed a benchmark with
**checkable answers**, and to be explicit about everything that cannot.

## Headline

**628 of 3,276 (19.2%)** are verifiable now — answerable today by querying
published aggregate tables. The other 80.8% are real research questions, but not
lookups.

## What kind of answer each question wants

| answer type | n | share |
|---|---:|---:|
| comparison | 816 | 24.9% |
| **qualitative** (mechanism, explanation) | 701 | 21.4% |
| trend (a series, usually a plot) | 499 | 15.2% |
| distribution (breakdown across categories) | 493 | 15.0% |
| yes_no | 173 | 5.3% |
| single_number | 138 | 4.2% |
| rate_or_share | 135 | 4.1% |
| ranking_or_list | 125 | 3.8% |
| normative (what policy should do) | 66 | 2.0% |
| other | 65 | 2.0% |
| methodological (about the study's own data) | 65 | 2.0% |

Only **8.3%** (`single_number` + `rate_or_share`) want a bare scalar. A
question-answering system for this audience that only returns numbers addresses
under a tenth of what researchers actually ask.

## What data would be needed

| | n | share |
|---|---:|---:|
| microdata (record-level, project licence) | 1,202 | 36.7% |
| other_source (survey, interviews, literature) | 1,054 | 32.2% |
| public_aggregate (StatLine / Eurostat) | 1,008 | 30.8% |
| none (not an empirical question) | 12 | 0.4% |

**The single largest bucket needs confidential microdata.** That is a finding
rather than a limitation: it quantifies the gap the public metadata layer cannot
close, and it is exactly the routing task worth building — telling a researcher
"this needs a CBS project licence, and here is the closest public proxy."

## Which publisher

| | n |
|---|---:|
| CBS | 1,998 |
| not_applicable | 1,230 |
| either | 41 |
| **ESTAT** | **7** |

The attested pool is essentially **CBS-only**. The corpus comes from Dutch
publications, so Eurostat questions barely occur. Any Eurostat arm of a benchmark
needs a different seed source — EU-level policy publications, Eurostat's own
"Statistics Explained" articles, or questions elicited directly from researchers.
Generating them from Eurostat metadata is possible but reintroduces the leakage
problem: such questions are written *from* the catalogue a retriever searches.

## The verifiable 628, by shape

| | n |
|---|---:|
| trend | 199 |
| distribution | 173 |
| comparison | 123 |
| single_number | 57 |
| rate_or_share | 44 |
| ranking_or_list | 25 |
| yes_no | 7 |

Publisher split: CBS 615 · either 8 · ESTAT 5.

**372 of the 628 want a series or a breakdown** — a plot or a table, not a
number. Scoring these needs a different protocol from exact match: comparing a
returned series against a reference series (shape, coverage, values within
tolerance), which is closer to table-QA evaluation than to QA exact match.

## The immediately usable set

Filtering to *verifiable* **and** *single-valued* **and** *from a document with at
least one dataset already linked to the catalogue*:

**14 questions.**

That is the honest size of the ready-made `(question, dataset, answer)` pool
today. Examples:

- *What is the number of procurement procedures in the Netherlands in the period
  2017–2019?* → `85951NED`, `84122NED`
- *Which regions showed the smallest decline or even stabilization in turnover?*
  → `81578NED` Vestigingen van bedrijven; bedrijfstak, regio
- *Did the post-WWII baby boom compensate for the births that did not occur
  during WWII?* → `85524NED` Bevolking, huishoudens en bevolkingsontwikkeling

The binding constraint is **not** the question supply — it is dataset linking.
252 verifiable single-valued CBS questions exist; only 25 come from a document
with any linked table, and only 14 survive both filters. Improving
mention→table matching turns hundreds of these into usable pairs.

## Output files

    data/processed/benchmark/seeds_cbs.jsonl        623  (252 single-valued, 25 linked)
    data/processed/benchmark/seeds_estat.jsonl        5  (4 single-valued, 0 linked)
    data/processed/benchmark/seeds_rejected.jsonl  2,648  each with `rejected_because`

Rejected questions are kept with their reason so the filter is auditable and can
be revised — several categories (`trend`, `distribution`) are excluded from the
*single-valued* set but are perfectly good benchmark items under a series-scoring
protocol.

## Caveats

- **These labels are model-assigned, not human-verified.** Temperature 0 and a
  constrained enum make them reproducible, not correct. Before this becomes gold,
  a human should check a sample — a few hundred items, with agreement reported.
- **`verifiable_now` is a judgement about the world, not the corpus.** The model
  is guessing whether a public table exists that answers the question. It has not
  checked the catalogue. A verification pass that actually attempts retrieval
  would be stricter and more useful.
- **`publisher_hint` for a Dutch question is close to a constant.** Its value is
  in flagging the rare cross-country question, not in partitioning the corpus.

*Regenerate:* `python -m enrich.classify_questions && python scripts/build_seed_sets.py`
