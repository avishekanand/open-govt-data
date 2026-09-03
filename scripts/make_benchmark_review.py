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
    items = [json.loads(l) for l in ITEMS.open(encoding="utf-8")] if ITEMS.exists() else []
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
    tierA = [i for i in items if i.get("executable")]
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
    triaged = [q for q in qs if q.get("benchmark_status")]
    if triaged:
        A("## 3. Candidate pool from triage")
        A("")
        A("| status | n |")
        A("|---|---:|")
        for k, v in Counter(q.get("benchmark_status") for q in triaged).most_common():
            A(f"| `{k}` | {v:,} |")
        A("")
        A("| specificity | n |")
        A("|---|---:|")
        for k, v in Counter(q.get("specificity") for q in triaged).most_common():
            A(f"| `{k}` | {v:,} |")
        A("")
        gold = [q for q in triaged if q.get("gold_ready")]
        A(f"### 3a. `gold_ready` candidates awaiting gold SQL ({len(gold)})")
        A("")
        A("Open data **and** a dataset attributed to this specific question **and**")
        A("specific enough to have one answer. These are next in line to be authored.")
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
                 if q.get("benchmark_status") == "open_data"
                 and q.get("specificity") == "underspecified"][:25]
        A(f"### 3b. Open data but underspecified — need a decision ({len(under)} shown)")
        A("")
        A("Answerable in principle, but a period, population or measure must be")
        A("pinned first. The `missing` column is what a reviewer would have to fix.")
        A("")
        A("| question | missing |")
        A("|---|---|")
        for q in under:
            A(f"| {clip(q.get('question_en'), 100)} | {clip(q.get('missing_to_specify'), 80)} |")
        A("")
        excl = [q for q in triaged if q.get("benchmark_status") == "not_a_data_question"][:15]
        A("### 3c. Excluded as not-a-data-question — spot-check the exclusions")
        A("")
        A("Listed so over-exclusion is visible. An earlier version of this pass")
        A("mislabelled vague-but-answerable questions here; see methodology §9.")
        A("")
        A("| question | reason |")
        A("|---|---|")
        for q in excl:
            A(f"| {clip(q.get('question_en'), 100)} | {clip(q.get('status_reason'), 80)} |")
        A("")
    else:
        A("## 3. Candidate pool from triage")
        A("")
        A("*Triage has not been run against the current corpus.*")
        A("")

    A("## How to review")
    A("")
    A("1. **Section 1** — check each verified item end to end. A wrong dataset or a")
    A("   query that answers a different question is the failure to look for.")
    A("2. **Section 3a** — check the question↔dataset pairing before SQL is written;")
    A("   this is where the earlier document-level attribution bug did its damage.")
    A("3. **Section 3c** — check we are not throwing away good questions.")
    A("")
    A("*Regenerate:* `python scripts/make_benchmark_review.py`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
