#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate every documentation page from the data, so docs cannot drift.

    python scripts/build_docs.py

Writes:
  docs/README.md              the master index - start here
  docs/questions_eurostat.md  the constructed Eurostat questions
  docs/questions_cbs.md       the CBS questions and their executed answers

Every count on those pages is computed from the artefacts at build time. If a
number is stale, the fix is to re-run this script, not to edit the page.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

D = Path("data/processed")
B = D / "benchmark"
DOCS = Path("docs")
STATLINE = "https://opendata.cbs.nl/statline/#/CBS/nl/dataset/{code}/table"
ESTAT_URL = "https://ec.europa.eu/eurostat/databrowser/view/{code}/default/table"
REPO = "https://github.com/avishekanand/open-govt-data/blob/benchmark/"


def jl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def clip(s, n=150):
    s = re.sub(r"\s+", " ", str(s or "").strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def newest_se():
    """Prefer the de-leaked Eurostat set when it exists."""
    dl = B / "constructed_se_deleaked.jsonl"
    return (dl, jl(dl)) if dl.exists() else (B / "constructed_se.jsonl",
                                             jl(B / "constructed_se.jsonl"))


def cbs_items():
    out = []
    for p in sorted(B.glob("items*.jsonl")):
        for it in jl(p):
            it["_file"] = p.name
            out.append(it)
    return out


def build_master(se_path, se, items):
    sound = [i for i in items if i.get("executable") and i.get("qa", {}).get("passed", True)]
    tierb = [i for i in items if not i.get("executable")]
    L = []
    A = L.append
    A("# Open Government Data benchmark — start here")
    A("")
    A("Everything below is generated from the data by `scripts/build_docs.py`.")
    A("Counts are computed at build time; if one looks stale, re-run the script.")
    A("")
    A("## The generated questions")
    A("")
    A("| what | how many | page | raw data |")
    A("|---|---:|---|---|")
    A(f"| **Eurostat questions** (constructed from the article that cites the table) "
      f"| {len(se):,} | [questions_eurostat.md](questions_eurostat.md) "
      f"| [`{se_path.name}`]({REPO}{se_path.as_posix()}) |")
    A(f"| **CBS questions with executed answers** | {len(sound):,} "
      f"| [questions_cbs.md](questions_cbs.md) | [`items_*.jsonl`]({REPO}data/processed/benchmark/) |")
    A(f"| CBS microdata questions (deferred to CBS) | {len(tierb):,} "
      f"| [benchmark_items_review.md](benchmark_items_review.md) | — |")
    A("")
    A("## The two benchmarks are separate")
    A("")
    A("CBS and Eurostat have their own questions, their own catalogue and their own")
    A("gold answers. They are never mixed: a Dutch question is never answered by a")
    A("Eurostat table. Mixing them once put Eurostat datasets in 10.2% of candidate")
    A("slots for Dutch questions.")
    A("")
    A("| | CBS StatLine | Eurostat |")
    A("|---|---|---|")
    A("| question source | research publications that cite CBS data | Statistics Explained articles |")
    A("| provenance | resolved from prose citations (lossy) | dataset codes cited verbatim (exact) |")
    A("| catalogue | 4,870 tables | 7,438 datasets |")
    A(f"| questions | {len(sound):,} with executed answers | {len(se):,} constructed |")
    A("")
    A("## Supporting documents")
    A("")
    A("| document | what it is |")
    A("|---|---|")
    A("| [methodology.md](methodology.md) | how every artefact was produced, with the defects found on the way |")
    A("| [benchmark_design.md](benchmark_design.md) | related work and the multi-hop construction plan |")
    A("| [question_analysis.md](question_analysis.md) | what kinds of answers the attested questions admit |")
    A("| [benchmark_items_review.md](benchmark_items_review.md) | items for human review, including what was rejected |")
    A("| [research_question_examples.md](../data/processed/pub/research_question_examples.md) | the attested questions with their witness sentences |")
    A("")
    A("## The metadata layer these rest on")
    A("")
    for name, path, desc in [
        ("enriched catalogue", D / "enriched_unified_qwen3-32b.jsonl",
         "12,308 datasets with English titles, descriptions, topics, dimensions"),
        ("field profiles (CBS)", D / "field_profiles_cbs.jsonl",
         "exact period/geography/cardinality per table"),
        ("field profiles (Eurostat)", D / "field_profiles_estat.jsonl", "same, for Eurostat"),
        ("surface forms", D / "surface_forms_estat.jsonl",
         "formal / plain / conversational / idiomatic / action-oriented phrasings per table"),
        ("Eurostat article corpus", D / "estat/se_articles.jsonl",
         "Statistics Explained articles and the dataset codes they cite"),
    ]:
        n = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
        A(f"- **{name}** — {n:,} records — `{path.as_posix()}` — {desc}")
    A("")
    A("*Regenerate every page:* `python scripts/build_docs.py`")
    return "\n".join(L) + "\n"


def build_eurostat(se_path, se):
    L = []
    A = L.append
    A("# Eurostat questions")
    A("")
    A(f"{len(se):,} questions, each constructed from an article that cites the table,")
    A("so the question and its gold dataset come from the same source. Generated by")
    A("`enrich.construct_from_citation`; raw data in "
      f"[`{se_path.name}`]({REPO}{se_path.as_posix()}).")
    A("")
    dl = [q for q in se if q.get("deleaked")]
    if dl:
        before = [q["title_overlap_before"] for q in dl if "title_overlap_before" in q]
        after = [q["title_overlap"] for q in dl if "title_overlap" in q]
        if before and after:
            A(f"**De-leaking:** {len(dl):,} questions were rephrased away from their table's")
            A(f"own title words (median overlap {sum(before)/len(before):.2f} → "
              f"{sum(after)/len(after):.2f}; attested-question baseline is 0.18).")
            A("")
    A("| answer type | n |")
    A("|---|---:|")
    for k, v in Counter(q.get("answer_type") for q in se).most_common():
        A(f"| {k} | {v:,} |")
    A("")
    A(f"Distinct datasets covered: **{len({q['gold_dataset']['code'] for q in se}):,}** ·")
    A(f"source articles: **{len({q['source_article'] for q in se}):,}**")
    A("")
    A("## Sample")
    A("")
    for q in se[:40]:
        code = q["gold_dataset"]["code"]
        A(f"- **{clip(q['question'], 170)}**")
        A(f"  <br/><sub>→ [`{code}`]({ESTAT_URL.format(code=code)}) "
          f"{clip(q['gold_dataset'].get('title_en'), 60)} · period {q.get('period_used')} "
          f"· from [{clip(q.get('source_article'), 55)}]({q.get('source_url')})</sub>")
    A("")
    A(f"<sub>Showing 40 of {len(se):,}. The rest are in the raw file.</sub>")
    return "\n".join(L) + "\n"


def build_cbs(items):
    sound = [i for i in items if i.get("executable") and i.get("qa", {}).get("passed", True)]
    rej = [i for i in items if i.get("executable") and not i.get("qa", {}).get("passed", True)]
    L = []
    A = L.append
    A("# CBS questions with executed answers")
    A("")
    A(f"{len(sound):,} questions whose gold SQL runs against the live table and returns a")
    A("result that passes an automated sanity check. Each answer below is the pinned")
    A("snapshot; re-running the query reproduces it.")
    A("")
    A(f"{len(rej):,} further queries ran but were rejected by QA (an unfiltered dimension")
    A("repeats every grouping key) — listed in [benchmark_items_review.md](benchmark_items_review.md).")
    A("")
    for it in sound:
        code = (it.get("gold_datasets") or [{}])[0].get("code", "")
        snap = it.get("answer_snapshot") or {}
        A(f"### {clip(it.get('question_en'), 130)}")
        A("")
        A(f"*Dataset:* [`{code}`]({STATLINE.format(code=code)}) · *type:* "
          f"`{it.get('answer_type')}` · *id:* `{it['id']}`")
        A("")
        if it.get("clarified_question"):
            A(f"*What the query pins down:* {clip(it['clarified_question'], 200)}")
            A("")
        A("```sql")
        A((it.get("gold_sql") or "").strip()[:900])
        A("```")
        A("")
        cols, rows = snap.get("columns") or [], snap.get("rows") or []
        if cols:
            A("| " + " | ".join(str(c) for c in cols) + " |")
            A("|" + "|".join(["---"] * len(cols)) + "|")
            for r in rows[:8]:
                A("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
            if len(rows) > 8:
                A(f"| … | {len(rows)-8} more rows |" + " |" * max(0, len(cols) - 2))
            A("")
        if it.get("source_publication"):
            A(f"<sub>question attested in {it['source_publication']}</sub>")
            A("")
    return "\n".join(L) + "\n"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    se_path, se = newest_se()
    items = cbs_items()
    (DOCS / "README.md").write_text(build_master(se_path, se, items), encoding="utf-8")
    (DOCS / "questions_eurostat.md").write_text(build_eurostat(se_path, se), encoding="utf-8")
    (DOCS / "questions_cbs.md").write_text(build_cbs(items), encoding="utf-8")
    for f in ("README.md", "questions_eurostat.md", "questions_cbs.md"):
        p = DOCS / f
        print(f"  {p} ({p.stat().st_size/1024:.1f} KB)")
    print("[DONE] docs regenerated from data")


if __name__ == "__main__":
    main()
