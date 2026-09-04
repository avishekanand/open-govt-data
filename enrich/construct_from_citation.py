#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construct answerable questions from an article's OWN cited table.

The retrieval-first pipeline loses almost everything at the verification step:
1,001 questions -> 136 with an answerable dataset. The loss is structural - it
takes a question written for one purpose and hunts for a table that happens to
match it.

This inverts that. The article already told us which table it used. So take
  * the article's thesis (what it is arguing),
  * the evidence line (the sentence where it cites the table), and
  * that table's REAL schema - dimensions, category values, coverage, measures
and ask for a question that the article's argument motivates AND the table can
answer, with the temporal and categorical constraints respected by construction
rather than checked afterwards.

Provenance is exact here: Statistics Explained articles cite dataset codes
verbatim, so there is no fuzzy matching and no citation pollution.

    python -m enrich.construct_from_citation --source se --dry-run
    python -m enrich.construct_from_citation --source se --model Qwen/Qwen3-32B-AWQ
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

SE = Path("data/processed/estat/se_articles.jsonl")
CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
OUT_TMPL = "data/processed/benchmark/constructed_{src}.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B-AWQ")

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "why_the_article_motivates_it": {"type": "string"},
        "period_used": {"type": "string"},
        "dimensions_to_use": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "measure_to_use": {"type": "string"},
        "answer_type": {"type": "string",
                        "enum": ["single_number", "rate_or_share", "comparison",
                                 "trend", "ranking_or_list", "distribution"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["question", "why_the_article_motivates_it", "period_used",
                 "dimensions_to_use", "measure_to_use", "answer_type", "confidence"],
    "additionalProperties": False,
}

SYSTEM = ("You write research questions that a specific statistical table can answer and "
          "that follow from what an article is arguing. You use only periods, categories "
          "and measures that the table actually has. Return STRICT JSON only.")

USER = """\
Article: {title}

What the article is about (opening):
\"\"\"
{thesis}
\"\"\"

Where it cites this table:
\"\"\"
{evidence}
\"\"\"

The cited table:
  {code} — {dtitle}
  Coverage: {coverage}
  Dimensions (with real category values):
{dims}
  Measures:
{measures}

Write ONE question that:
  1. follows from what this article is investigating - it should be a question a
     reader of the article would want answered, not a description of the table;
  2. can be answered from THIS table alone, using only the periods, categories
     and measures listed above;
  3. respects the table's constraints by construction: name a period the table
     actually covers, and breakdowns it actually has.

Do not copy the table's title wording. Write it as a researcher would ask it.

Return JSON:
{{
  "question": "<the question>",
  "why_the_article_motivates_it": "<one sentence linking it to the article>",
  "period_used": "<a period the table covers>",
  "dimensions_to_use": ["<dimension ids>"],
  "measure_to_use": "<measure>",
  "answer_type": "single_number | rate_or_share | comparison | trend | ranking_or_list | distribution",
  "confidence": "high | medium | low"
}}
"""


def fmt_dims(rec, max_cats=10):
    out = []
    for d in (rec.get("dimensions") or [])[:8]:
        s = ", ".join(str(x) for x in (d.get("sample") or [])[:max_cats])
        out.append(f"    - {d.get('id')} ({d.get('n_categories')} categories): {s}")
    return "\n".join(out) or "    (none)"


def fmt_measures(rec):
    ms = rec.get("measures") or []
    return "\n".join(f"    - {m.get('name')} [{m.get('unit') or ''}]" for m in ms[:8]) \
        or "    (single value column; unit is a dimension)"


def evidence_window(text: str, code: str, width: int = 700) -> str:
    """The passage where the article cites this dataset code."""
    m = re.search(re.escape(code), text, re.I)
    if not m:
        return ""
    a = max(0, m.start() - width // 2)
    return re.sub(r"\s+", " ", text[a: m.start() + width // 2]).strip()


def clean_lead(text: str, n: int = 1200) -> str:
    """Article opening, with wiki markup stripped enough to read."""
    t = re.sub(r"\{\{[^}]*\}\}", " ", text)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"'{2,}|={2,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Construct questions from cited tables")
    ap.add_argument("--source", choices=["se"], default="se")
    ap.add_argument("--corpus", default=CORPUS, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-per-article", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out = args.out or Path(OUT_TMPL.format(src=args.source))
    corpus = {}
    with args.corpus.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            corpus[r["code"].lower()] = r

    jobs = []
    with SE.open(encoding="utf-8") as fh:
        for line in fh:
            a = json.loads(line)
            codes = a.get("codes_in_catalogue") or []
            if not codes:
                continue
            lead = clean_lead(a["text"])
            for code in codes[: args.max_per_article]:
                rec = corpus.get(code.lower())
                if not rec:
                    continue
                jobs.append({"article": a["title"], "url": a["url"], "code": code,
                             "rec": rec, "lead": lead,
                             "evidence": evidence_window(a["text"], code)})
    if args.limit:
        jobs = jobs[: args.limit]
    arts = len({j["article"] for j in jobs})
    print(f"[INFO] {len(jobs):,} (article, cited table) pairs from {arts:,} articles")

    def prompt(j):
        rec = j["rec"]
        cov = rec.get("coverage") or {}
        return USER.format(
            title=j["article"], thesis=j["lead"], evidence=j["evidence"] or "(not located)",
            code=rec["code"], dtitle=rec.get("title_en") or rec.get("title_native"),
            coverage=(f"{cov.get('start')} to {cov.get('end')}" if cov.get("start") else "(unknown)"),
            dims=fmt_dims(rec), measures=fmt_measures(rec))

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if jobs:
            print("\n--- sample prompt ---\n" + prompt(jobs[0])[:2000])
        return
    if not jobs:
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(j)}]
             for j in jobs]
    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for j, o in zip(jobs, outs):
            obj = parse_json(o.outputs[0].text if o.outputs else "")
            if not obj or not obj.get("question"):
                continue
            ok += 1
            fh.write(json.dumps({
                "question": obj["question"],
                "answer_type": obj.get("answer_type"),
                "period_used": obj.get("period_used"),
                "dimensions_to_use": obj.get("dimensions_to_use"),
                "measure_to_use": obj.get("measure_to_use"),
                "why_the_article_motivates_it": obj.get("why_the_article_motivates_it"),
                "confidence": obj.get("confidence"),
                "gold_dataset": {"code": j["rec"]["code"], "publisher": j["rec"].get("publisher"),
                                 "title_en": j["rec"].get("title_en")},
                "provenance": "cited_by_article",
                "source_article": j["article"], "source_url": j["url"],
                "evidence_line": j["evidence"][:400],
            }, ensure_ascii=False) + "\n")
    print(f"[DONE] constructed {ok:,} questions -> {args.out}")


if __name__ == "__main__":
    main()
