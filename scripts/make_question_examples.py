#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render a browsable Markdown sample of the extracted research questions.

    python scripts/make_question_examples.py

Input : data/processed/pub/pub_evidence.jsonl   (from enrich.pub_evidence)
Output: data/processed/pub/research_question_examples.md
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC = Path("data/processed/pub/pub_evidence.jsonl")
OUT = Path("data/processed/pub/research_question_examples.md")
STATLINE = "https://opendata.cbs.nl/statline/#/CBS/nl/dataset/{code}/table"

THEMES = [
    ("Labour, income and social security", r"arbeid|werk|inkomen|uitkering|loon|pensioen|armoede|bijstand"),
    ("Health and care", r"zorg|gezond|ziekte|patiënt|sterfte|overleden|ggz"),
    ("Education and youth", r"onderwijs|student|school|jeugd|leerling|studie"),
    ("Housing, regions and liveability", r"woning|huur|wonen|buurt|wijk|leefbaar|gemeente|regio"),
    ("Migration, population and diversity", r"migra|bevolking|herkomst|vergrijzing|demograf|integratie"),
    ("Business, economy and innovation", r"bedrijf|onderneming|economie|innovatie|export|mkb"),
]


def src(r):
    t = re.sub(r"\s+", " ", (r.get("title") or "").strip())
    if not t or len(t) < 4:
        t = (r.get("domain") or r["url"])[:70]
    return t[:95], r.get("final_url") or r["url"]


def clip(s, n=210):
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= n else s[: n - 1] + "…"


def linked(r):
    return {x["link"]["code"]: (x["link"]["title"] or "")
            for x in r["data_mentions"] if x["link"]["match"] in ("exact_id", "fuzzy_title")}


def registers(r):
    return sorted({x["link"]["code"] for x in r["data_mentions"] if x["link"]["match"] == "register"})


def build() -> str:
    recs = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    q = [x for r in recs for x in r["research_questions"]]
    m = [x for r in recs for x in r["data_mentions"]]
    L: list[str] = []
    A = L.append

    A("# Research questions extracted from publications")
    A("")
    A("Real questions that published studies set out to answer, recovered automatically")
    A("from the full text of the publications themselves — not generated from metadata.")
    A("")
    A("**How these were produced.** Every publication in the CBS publications workbook")
    A("was crawled (`cbs.pub_download`), its complete text extracted, and each chunk")
    A("carrying a CBS/Eurostat signal — plus the opening chunks, where a paper states")
    A("its aim — passed to Qwen3-32B under a constrained JSON schema")
    A("(`enrich.pub_evidence`). For every question the model must quote a **witness")
    A("sentence verbatim from the document**; that quote is then checked to actually")
    A("occur in the source text and flagged if it does not. Dataset mentions are")
    A("resolved against the 12,308-dataset catalogue by `enrich.pub_link`.")
    A("")
    A("Questions appear in their original language (mostly Dutch).")
    A("")
    A("| | |")
    A("|---|---|")
    A(f"| documents processed | {len(recs):,} |")
    A(f"| research questions extracted | {len(q):,} |")
    A(f"| documents yielding ≥1 question | {sum(1 for r in recs if r['research_questions']):,} |")
    A(f"| questions with a verified witness | {sum(1 for x in q if x['witness_verified']) / len(q):.1%} |")
    A(f"| dataset mentions | {len(m):,} |")
    A("| model | Qwen3-32B (vLLM, schema-constrained) |")
    A("")
    A("---")
    A("")
    A("## Part 1 — Questions with the datasets the study used")
    A("")
    A("The benchmark-shaped cases: a real research question, plus the CBS tables the")
    A("publication actually cited, resolved to catalogue codes. A retrieval system")
    A("should be able to get from the question to those tables.")
    A("")

    both = sorted((r for r in recs if r["research_questions"] and linked(r)),
                  key=lambda r: -len(linked(r)))
    n = 0
    for r in both:
        qs = [x for x in r["research_questions"] if x["witness_verified"]] or r["research_questions"]
        if not qs:
            continue
        title, url = src(r)
        n += 1
        A(f"### {n}. {title}")
        A("")
        A(f"*Source:* <{url}>")
        A("")
        for x in qs[:3]:
            A(f"- **Q:** {clip(x['question'], 180)}")
            if x["witness"]:
                flag = "" if x["witness_verified"] else "  ⚠️ not verified in source"
                A(f"  - *witness:* “{clip(x['witness'], 200)}”{flag}")
        A("")
        A("  **Datasets used** (resolved to the catalogue):")
        A("")
        for code, t in list(linked(r).items())[:6]:
            A(f"  - [`{code}`]({STATLINE.format(code=code)}) — {clip(t, 70)}")
        if registers(r):
            A(f"  - *microdata registers (no public table):* {', '.join(registers(r))}")
        A("")
        if n >= 8:
            break

    A("---")
    A("")
    A("## Part 2 — A wider sample of extracted questions")
    A("")
    A("Every question below carries a witness sentence verified against the document")
    A("text. Grouping is a rough heuristic on keywords in the *source publication's")
    A("title*, not on the question itself, so an occasional question sits under a")
    A("neighbouring theme.")
    A("")
    used: set[str] = set()
    for name, pat in THEMES:
        rows = []
        for r in recs:
            title, url = src(r)
            if not re.search(pat, (title + " " + (r.get("domain") or "")).lower()):
                continue
            for x in r["research_questions"]:
                if not x["witness_verified"]:
                    continue
                key = x["question"][:60].lower()
                if key in used:
                    continue
                used.add(key)
                rows.append((x["question"], title, url))
                break
            if len(rows) >= 5:
                break
        if not rows:
            continue
        A(f"### {name}")
        A("")
        for qq, title, url in rows:
            A(f"- {clip(qq, 190)}")
            A(f"  <br/><sub>— [{clip(title, 72)}]({url})</sub>")
        A("")

    A("---")
    A("")
    A("## Caveats")
    A("")
    A("- **Witness verification is a floor, not a guarantee.** It proves the sentence")
    A("  exists in the document; it does not prove the model read it correctly. About")
    A("  9% of questions carry a witness that could not be located; those are excluded")
    A("  from Part 2 and flagged in Part 1.")
    A("- **Not every 'question' is a research question.** The workbook includes")
    A("  parliamentary documents (*Kamerstukken*), whose numbered `Vraag 1…` items are")
    A("  parliamentary questions to a minister rather than a study's research aim.")
    A("  They are genuine questions answered with CBS data, but a different genre.")
    A("- **Dataset linking resolves ~10% of mentions.** Most of the rest are microdata")
    A("  registers, surveys and CBS statistical programmes with no StatLine table id.")
    A("  The addressable gap is tables named in prose that lexical matching misses on")
    A("  vocabulary; embedding matching over the enriched English titles would close")
    A("  much of it.")
    A("- **Questions are chunk-scoped** — taken from the opening of each document,")
    A("  where the aim is normally stated. A question buried later may be missed.")
    A("- **Not every document is CBS-related.** 505 of 1,176 were classified as using")
    A("  no CBS data.")
    A("")
    A("*Regenerate:* `python -m enrich.pub_evidence --resume && "
      "python scripts/make_question_examples.py`")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
