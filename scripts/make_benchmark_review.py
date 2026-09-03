#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the benchmark items and candidates as reviewable Markdown.

    python scripts/make_benchmark_review.py

Inputs : data/processed/benchmark/items_v0.jsonl   (authored items + snapshots)
         data/processed/pub/pub_evidence.jsonl     (triaged questions)
Output : docs/benchmark_items_review.md
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ITEMS = Path("data/processed/benchmark/items_v0.jsonl")
CANDS = Path("data/processed/benchmark/question_dataset_candidates.jsonl")
EVID = Path("data/processed/pub/pub_evidence.jsonl")
OUT = Path("docs/benchmark_items_review.md")
STATLINE = "https://opendata.cbs.nl/statline/#/CBS/nl/dataset/{code}/table"


def clip(s, n=180):
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def table(cols, rows, limit=12):
    L = ["| " + " | ".join(str(c) for c in cols) + " |",
         "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows[:limit]:
        L.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    if len(rows) > limit:
        L.append(f"| … | {len(rows) - limit} more rows |" + " |" * (len(cols) - 2))
    return L


def build() -> str:
    L: list[str] = []
    A = L.append
    import glob
    items = []
    for f in sorted(glob.glob("data/processed/benchmark/items*.jsonl")):
        items += [json.loads(l) for l in open(f, encoding="utf-8")]
    recs = [json.loads(l) for l in EVID.open(encoding="utf-8")] if EVID.exists() else []
    qs = [q for r in recs for q in (r.get("research_questions") or [])]

    A("# Benchmark items — for review")
    A("")
    A("Everything below is **draft and unvalidated**. The point of this page is to")
    A("let a domain expert check the question, the dataset it was paired with, the")
    A("query, and the answer — and reject any of them. Method: "
      "[methodology.md](methodology.md).")
    A("")

    # ---------------------------------------------------------- authored items
    tierA = [i for i in items if i.get("executable") and i.get("qa", {}).get("passed", True)]
    rejected = [i for i in items if i.get("executable") and not i.get("qa", {}).get("passed", True)]
    tierB = [i for i in items if not i.get("executable")]
    A(f"## 1. Verified items — Tier A ({len(tierA)})")
    A("")
    A("Gold SQL executed against the live table; the answer below is the pinned")
    A("snapshot. Re-running the query reproduces it, and a perturbed query fails.")
    A("")
    for it in tierA:
        A(f"### `{it['id']}` — {clip(it.get('question_en'), 150)}")
        A("")
        if it.get("question_native"):
            A(f"*Original:* {clip(it['question_native'], 150)}")
            A("")
        if it.get("clarified_question"):
            A(f"*Clarified (what the query actually pins down):* {clip(it['clarified_question'], 220)}")
            A("")
        A(f"*Answer type:* `{it.get('answer_type')}` · *publisher:* {it.get('publisher')}")
        A("")
        A("**Dataset(s):**")
        for d in it.get("gold_datasets") or []:
            A(f"- [`{d['code']}`]({STATLINE.format(code=d['code'])}) — {clip(d.get('title'), 80)}")
        A("")
        A("**Gold SQL**")
        A("")
        A("```sql")
        A(it.get("gold_sql", "").strip())
        A("```")
        A("")
        snap = it.get("answer_snapshot") or {}
        if snap.get("columns"):
            A("**Answer**")
            A("")
            L.extend(table(snap["columns"], snap.get("rows") or []))
            A("")
            A(f"<sub>retrieved {snap.get('retrieved_at')} · table last updated "
              f"{snap.get('table_last_update')}</sub>")
            A("")
        if it.get("source_publication"):
            A(f"*Question attested in:* <{it['source_publication']}>")
            A("")
        A("**Review:** is the question well-posed? is this the dataset you would use? "
          "does the SQL express the question? is the answer right?")
        A("")
        A("---")
        A("")

    A(f"## 1b. Rejected by automated QA ({len(rejected)})")
    A("")
    A("These queries ran and returned rows, but the result cannot be right: a")
    A("dimension was left unfiltered, so each grouping key repeats with different")
    A("values. Shown so the failure mode is visible rather than hidden.")
    A("")
    if rejected:
        A("| id | question | why rejected |")
        A("|---|---|---|")
        for it in rejected:
            A(f"| `{it['id']}` | {clip(it.get('question_en'), 80)} | "
              f"{clip(it.get('qa', {}).get('reason'), 70)} |")
        A("")

    A(f"## 2. Deferred items — Tier B, microdata ({len(tierB)})")
    A("")
    A("Gold SQL is authored against the documented CBS register schemas and is")
    A("**not executable** outside the CBS secure environment. Kept so the microdata")
    A("questions stay in the benchmark as a *write the query you would run* task,")
    A("to be validated with CBS.")
    A("")
    for it in tierB:
        A(f"### `{it['id']}` — {clip(it.get('question_en'), 150)}")
        A("")
        A(f"*Registers:* {', '.join(it.get('microdata_registers') or []) or '—'}")
        A("")
        A("```sql")
        A(it.get("gold_sql", "").strip())
        A("```")
        A("")
        A("---")
        A("")

    # ------------------------------------------------------- triage candidates
    # NOTE: the triage `benchmark_status` field is NOT used. Two runs of it were
    # miscalibrated (73% then 91% of questions dumped into not_a_data_question,
    # including 867 the classifier had called public_aggregate). What survived
    # review is the per-question dataset ATTRIBUTION, which invented no codes.
    # Data availability therefore comes from the classifier pass instead.
    triaged = [q for q in qs if q.get("attributed_dataset") or q.get("data_needed")]
    if triaged:
        A("## 3. Candidate pool from triage")
        A("")
        A("Data availability is taken from the **classifier** pass (`data_needed`).")
        A("The triage `benchmark_status` field is deliberately ignored — see")
        A("[methodology §9](methodology.md#9-known-defects-and-how-they-were-caught).")
        A("")
        A("| data needed (classifier) | n |")
        A("|---|---:|")
        for k, v in Counter(q.get("data_needed") for q in triaged).most_common():
            A(f"| `{k}` | {v:,} |")
        A("")
        gold = [q for q in triaged
                if q.get("attributed_dataset") and q.get("data_needed") == "public_aggregate"
                and q.get("verifiable_now")]
        A(f"### 3a. Candidates awaiting gold SQL ({len(gold)})")
        A("")
        A("Three independent signals agree: the classifier called it answerable from")
        A("public aggregates, it marked it verifiable, and triage attributed a specific")
        A("dataset **to this question** (not merely to its publication). These are next")
        A("in line to be authored — and the first thing worth a human check.")
        A("")
        if gold:
            A("| question | dataset | confidence |")
            A("|---|---|---|")
            for q in gold[:40]:
                code = q.get("attributed_dataset")
                link = f"[`{code}`]({STATLINE.format(code=code)})" if code else "—"
                A(f"| {clip(q.get('question_en'), 110)} | {link} | {q.get('attribution_confidence')} |")
        else:
            A("*(none yet)*")
        A("")
        under = [q for q in triaged
                 if q.get("data_needed") == "public_aggregate"
                 and not q.get("attributed_dataset")
                 and q.get("verifiable_now")][:25]
        A(f"### 3b. Answerable from open data but no dataset attributed ({len(under)} shown)")
        A("")
        A("The classifier judged these answerable from published tables, but no")
        A("dataset could be attributed — because the source publication cited none")
        A("that we could link. Retrieval, not question quality, is the blocker.")
        A("")
        A("| question | what would have to be pinned |")
        A("|---|---|")
        for q in under:
            A(f"| {clip(q.get('question_en'), 100)} | {clip(q.get('missing_to_specify'), 80)} |")
        A("")
        excl = [q for q in triaged if q.get("data_needed") == "microdata"][:15]
        A("### 3c. Microdata questions — the Tier B pool")
        A("")
        A("The largest single bucket (1,202). Not answerable from public tables;")
        A("candidates for gold SQL authored against register schemas and validated")
        A("later with CBS.")
        A("")
        A("| question | classifier reason |")
        A("|---|---|")
        for q in excl:
            A(f"| {clip(q.get('question_en'), 100)} | {clip(q.get('reason'), 80)} |")
        A("")
    else:
        A("## 3. Candidate pool from triage")
        A("")
        A("*Triage has not been run against the current corpus.*")
        A("")

    # ------------------------------------------------- retrieval adjudication
    if CANDS.exists():
        rows = [json.loads(l) for l in CANDS.open(encoding="utf-8")]
        both = [r for r in rows if r.get("lexically_attributed")]
        A("## 4. Two candidate sources disagree — please adjudicate")
        A("")
        A("Two independent ways of finding a dataset for a question:")
        A("")
        A("- **cited** — the dataset the *publication itself named*, resolved lexically.")
        A("  Provenance-faithful: right if the task is *recompute the paper's number*.")
        A("- **retrieved** — nearest datasets by embedding over all 12,308 enriched")
        A("  catalogue entries. Task-faithful: right if the task is *find the data that")
        A("  answers this question*.")
        A("")
        A("They agree on **none** of the cases below. That is not a retrieval failure:")
        A("papers cite reference tables (region definitions, classifications) alongside")
        A("the table carrying the measure, so 'what the paper cited' and 'what answers")
        A("the question' genuinely differ. **Which one is gold is a decision, not a")
        A("computation** — hence this section.")
        A("")
        for r in both:
            A(f"**Q:** {clip(r.get('question_selfcontained') or r.get('question'), 190)}")
            A("")
            sc = r.get("scope") or {}
            if any(sc.values()):
                A("<sub>scope: " + " · ".join(f"{k}={v}" for k, v in sc.items() if v) + "</sub>")
                A("")
            A("| source | dataset | title |")
            A("|---|---|---|")
            lex = r["lexically_attributed"]
            A(f"| **cited** | [`{lex}`]({STATLINE.format(code=lex)}) | *(named in the publication)* |")
            for c in (r.get("candidates") or [])[:3]:
                A(f"| retrieved #{c['rank']} ({c['score']}) | [`{c['code']}`]"
                  f"({STATLINE.format(code=c['code'])}) | {clip(c.get('title_en'), 60)} |")
            A("")
            A("*Pick one, both, or neither.*")
            A("")
        sample = [r for r in rows if not r.get("lexically_attributed")][:20]
        A(f"### 4a. Retrieval-only candidates — sample of {len(sample)}")
        A("")
        A("Questions whose publication cited nothing we could link. These are the")
        A("~1,000 that lexical matching lost entirely; retrieval gives them a")
        A("candidate for the first time. Judge whether the top hit is usable.")
        A("")
        A("| question | top retrieved | score |")
        A("|---|---|---|")
        for r in sample:
            c = (r.get("candidates") or [{}])[0]
            code = c.get("code")
            link = f"[`{code}`]({STATLINE.format(code=code)})" if code else "—"
            A(f"| {clip(r.get('question_selfcontained') or r.get('question'), 95)} | "
              f"{link} {clip(c.get('title_en'), 40)} | {c.get('score')} |")
        A("")

    A("## How to review")
    A("")
    A("1. **Section 1** — check each verified item end to end. A wrong dataset or a")
    A("   query that answers a different question is the failure to look for.")
    A("2. **Section 3a** — check the question↔dataset pairing before SQL is written;")
    A("   this is where the earlier document-level attribution bug did its damage.")
    A("3. **Section 3c** — check we are not throwing away good questions.")
    A("4. **Section 4** — decide which notion of gold we want: the dataset the paper")
    A("   cited, or the dataset that best answers the question. This choice shapes the")
    A("   whole benchmark and cannot be settled automatically.")
    A("")
    A("*Regenerate:* `python scripts/make_benchmark_review.py`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
