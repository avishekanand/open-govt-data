# Designing a multi-hop benchmark over open government data

Working notes on how to turn the extracted publication questions and the enriched
dataset catalogue into a multi-hop retrieval/agent benchmark — and how to keep it
honest. Written for whoever picks this up next.

## 1. What we already have

| asset | size | provenance |
|---|---|---|
| enriched dataset catalogue | 12,308 datasets (7,438 Eurostat + 4,870 CBS) | metadata + LLM enrichment |
| real research questions | 3,276 from 1,176 publications | extracted full-text, witness-verified |
| dataset mentions | 3,329, ~10% linked to catalogue codes | LLM + lexical linking |
| generated questions | ~54k `example_queries`, 98.6% unique | generated **from** metadata |

The distinction that governs everything below: the 3,276 publication questions are
**attested** — a real study asked them. The ~54k `example_queries` are **synthetic**
— generated from the very metadata a retriever will search. They are useful as
paraphrase material and as distractors, but they must not become gold questions:
they leak their own answer's vocabulary.

## 2. Related work, and what to borrow from each

### Composition rather than generation
**MuSiQue** ([TACL 2022](https://aclanthology.org/2022.tacl-1.31/)) is the most
transferable method. It builds k-hop questions *bottom-up* by composing verified
single-hop questions, requiring that hop 2 genuinely depends on hop 1's answer,
then filters aggressively for shortcuts. The payoff is measurable: a human–model
gap of **28.2 F1**, versus 9.6 on HotpotQA and 3.7 on 2WikiMultiHopQA — evidence
that the earlier benchmarks were largely solvable *without* composing anything.

Borrow: bottom-up composition, explicit connectivity, shortcut filtering as a
first-class construction step rather than a post-hoc analysis.

### Templated composition over a graph
**2WikiMultiHopQA** composes Wikidata paths into questions via templates. Scales
cheaply and gives exact evidence chains, but phrasing is unnatural and models
learn the template. We have a natural-language corpus to anchor phrasing, so use
templates for the *reasoning structure* and real questions for the *surface form*.

### Adversarial hand-writing
**Bamboogle** is small and hand-written specifically so that retrieval shortcuts
fail. Worth reserving a few hundred hand-checked items as a hard slice, rather
than trying to make the whole benchmark adversarial.

### Bottom-up from semi-structured sources
[**BMGQ**](https://arxiv.org/html/2510.24151) generates complex multi-hop
questions bottom-up from semi-structured data — the closest published setting to
ours, where the "documents" are tables with schemas rather than prose.

### Dataset discovery is its own literature
Our core task — *which datasets answer this, and on what keys do they join* — is
table union/join search, and it has established benchmarks and metrics:

- [**LakeBench**](https://www.vldb.org/pvldb/vol17/p1925-chai.pdf) (VLDB 2024):
  join & union discovery over 16M tables, >10k queries, ~7,500 human labelling
  hours. Take its query taxonomy and its separation of effectiveness / efficiency
  / scalability.
- **TUS** (table union search) and **SANTOS** (KG-based semantic unionability):
  the attribute-level matching baselines any dataset-discovery system must beat.
- [**Evaluating joinable column discovery**](https://arxiv.org/pdf/2510.24599):
  context-aware join discovery evaluation — directly relevant to scoring the
  *join key* half of an answer, not just the dataset set.
- [**LakeQA**](https://arxiv.org/pdf/2606.10460): exploratory QA over a
  million-scale data lake — the QA-over-discovery framing.
- [**Generative benchmark creation for table union search**](https://arxiv.org/pdf/2308.03883):
  LLM-in-the-loop benchmark construction with human verification — a cautionary
  and practical reference for exactly what we are proposing.
- [**Survey: open dataset search in the LLM era**](https://arxiv.org/pdf/2509.00728):
  positioning for the retrieval side.

**The gap we can claim.** Multi-hop QA benchmarks compose over *text*; dataset
discovery benchmarks evaluate *retrieval* without a downstream answer. Nobody has
a benchmark where the hops are **datasets**, the join is part of the gold label,
the answer is **executable against live official statistics**, and the questions
are **attested by real publications**. That is the contribution.

## 3. Threat model — how this goes wrong

1. **Shortcut / disconnected reasoning.** The question is answerable from one
   dataset. The dominant failure in HotpotQA-style benchmarks.
2. **Metadata leakage.** The question reuses the gold dataset's title words, so
   BM25 alone solves retrieval. Acute here, because `example_queries` were
   generated *from* those titles.
3. **Unnatural composition.** "What was the unemployment rate in the region with
   the highest GDP per inhabitant?" is grammatical, composable — and nobody asks
   it. Templated composition produces these by default.
4. **Non-reproducible answers.** Official statistics get revised; a table that
   said 2023 last year says 2025 now (we hit exactly this). Unpinned answers rot.
5. **Definitional mismatch.** CBS and Eurostat measure "unemployment" and
   "household" differently. A cross-source hop can be arithmetically fine and
   semantically wrong.
6. **Genre contamination.** Our question pool mixes research questions with
   parliamentary *Kamerstuk* questions. Different distribution; label them.

## 4. Proposed procedure

**Stage 0 — seed pool.** Start only from attested questions: the 3,276 extracted
questions, each carrying a witness sentence, a source document and (where linked)
the datasets that study used. Keep the genre label.

**Stage 1 — anchor each seed to an executable single hop.** A seed is admitted
only if it can be expressed as `(dataset code, filters, measure) → value`, run
against the live API. Unanchored seeds go to a "discovery-only" pool (still
useful for retrieval evaluation, no numeric answer).

**Stage 2 — build the join graph.** Nodes are datasets; edges are verifiable
relations from metadata: shared dimension id, overlapping codelists (the same
`geo` codes), shared period coverage, and a curated NUTS↔CBS region bridge.
Edges must be checked against actual category values, never assumed from names.

**Stage 3 — compose along edges only.** A 2-hop item requires a *bridge entity*
produced by hop 1 and consumed by hop 2 (a region, a year, a sector). Never ask
an LLM for "a multi-hop question" — that is how unnatural and shortcut items get
in. Phrase the composed question by **retrieving the nearest attested seed
questions and imitating their register**, so surface form stays realistic.

**Stage 4 — automated filters (all must pass).**
- *Single-dataset solvability*: a strong model with retrieval over one dataset
  must fail. If it succeeds, discard. (MuSiQue's central check.)
- *Hop necessity*: ablate each gold dataset in turn; the item must become
  unanswerable. Catches decorative hops.
- *Leakage*: cap lexical overlap between question and gold dataset title/
  `title_en`; prefer publication vocabulary over catalogue vocabulary.
- *Executability*: the answer must be computable, with the data version pinned
  (`last_update` + retrieval date recorded).

**Stage 5 — human validation with social scientists.** The step that decides
whether this is real. For a sample: *would you ask this?* (Likert), *is it
well-posed?*, *are the gold datasets the ones you would use?* Report
inter-annotator agreement, and keep the raters' free-text objections — they are
the most informative artefact for revising templates.

**Stage 6 — release with full provenance.** Per item: question (nl + en), hop
decomposition, gold dataset codes, join keys, executable program, answer, version
pin, source publication URL, witness sentence, genre label, and every filter's
verdict. Provenance is what makes the benchmark auditable rather than assertable.

## 5. Hop templates grounded in this corpus

| # | pattern | example shape | needs |
|---|---|---|---|
| 1 | bridge by region | X by municipality → the extreme region → Y there | shared `geo` codelist |
| 2 | cross-source bridge | Eurostat NUTS-3 GDP → matching CBS region → CBS indicator | NUTS↔CBS map |
| 3 | temporal bridge | year of an event from the publication → value that year | period coverage |
| 4 | fan-out / aggregation | same measure across N categories, then compare | one dataset, many cells |
| 5 | provenance hop | which datasets did study P use → recompute its number | pub_evidence links |
| 6 | definitional reconciliation | same concept, two sources, different definitions | both catalogues |

Types 1–3 are genuine multi-hop retrieval. Type 4 is fan-out (à la aggregation
benchmarks) and should be reported separately — it is one retrieval, many reads.
Type 5 is the one this corpus uniquely enables. Type 6 is the hardest and the
most useful to social scientists.

## 6. Metrics

- **Discovery**: recall@k / MRR over the gold dataset *set* (not a single doc).
- **Join correctness**: accuracy of predicted join keys — scored separately, per
  the joinable-column-discovery literature.
- **Answer**: exact match for counts, tolerance band for rates and ratios.
- **Faithfulness**: did the agent actually read all gold hops, or guess?
- **Shortcut rate**: fraction solvable single-hop. Report it every release; it is
  the number that ages a benchmark.

## 7. Known limits today

- **Linking is the bottleneck.** Only ~49 distinct catalogue tables are currently
  linked from publications — far too thin for gold labels at scale. Embedding
  matching over enriched English titles is the prerequisite, not an optimisation.
- **Skew.** The pool is Dutch policy research; conclusions will not transfer
  automatically to other national statistical offices.
- **No microdata.** Many attested questions need CBS microdata access. Those
  become *routing* items ("this is not publicly answerable, and here is why"),
  which is a legitimate and useful task in itself.

## Sources

- MuSiQue — https://aclanthology.org/2022.tacl-1.31/
- BMGQ — https://arxiv.org/html/2510.24151
- LakeBench — https://www.vldb.org/pvldb/vol17/p1925-chai.pdf
- LakeQA — https://arxiv.org/pdf/2606.10460
- Joinable column discovery — https://arxiv.org/pdf/2510.24599
- Generative benchmark creation for table union search — https://arxiv.org/pdf/2308.03883
- Survey: open dataset search in the LLM era — https://arxiv.org/pdf/2509.00728
