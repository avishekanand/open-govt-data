#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Derive answerable questions that stay in the spirit of the article's question.

So far the pipeline has been one-directional: take the article's question
verbatim, then ask whether some dataset can answer it. Most of the time the
answer is no - not because the question is bad but because the exact period,
region or breakdown the article used is not in a public table.

This closes the loop the other way. Given the article's research question AND a
dataset's real schema, write a question that
  (a) can actually be answered from that dataset, and
  (b) still serves what the article set out to find out.

The article's question is never discarded: each derived question records the
original, what was changed, and a faithfulness judgement, so a reviewer can see
exactly how far it drifted.

Leakage control: the derived question must be phrased in the article's own
register and must NOT copy the dataset's title wording, otherwise the benchmark
becomes trivially solvable by lexical matching against the catalogue. Overlap is
measured afterwards by scripts, not trusted to the prompt.

    python -m enrich.derive_questions --dry-run
    python -m enrich.derive_questions --model Qwen/Qwen3-32B
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

SRC = Path("data/processed/benchmark/question_dataset_verified_cbs_relaxed.jsonl")
CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
OUT = Path("data/processed/benchmark/derived_questions_cbs.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")

SCHEMA = {
    "type": "object",
    "properties": {
        "derived_question": {"type": "string"},
        "what_changed": {"type": "string"},
        "serves_original": {"type": "string", "enum": ["yes", "partly", "no"]},
        "why_it_serves": {"type": "string"},
        "period_used": {"type": "string"},
        "dimensions_to_use": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "measure_to_use": {"type": "string"},
    },
    "required": ["derived_question", "what_changed", "serves_original", "why_it_serves",
                 "period_used", "dimensions_to_use", "measure_to_use"],
    "additionalProperties": False,
}

SYSTEM = ("You adapt a research question so that it can be answered from a specific "
          "statistical table, without losing what the researcher wanted to find out. "
          "Return STRICT JSON only.")

USER = """\
A publication asked this research question:
  "{q}"
Its stated scope: {scope}
Publication: {title}

The following table is topically right but does not match the question exactly:

Table {code} — {dtitle}
Coverage: {coverage}
Dimensions (with real category values):
{dims}
Measures:
{measures}

Write ONE question that:
  1. CAN be answered from this table alone - only periods, categories and
     measures shown above;
  2. still serves what the publication wanted to find out. Keep the subject and
     the kind of comparison. Moving the period, narrowing the population, or
     using the nearest available breakdown is fine. Changing the topic is not.

Phrase it the way the publication would - a researcher's question, not a
description of the table. Do NOT copy the table's title wording.

Return JSON:
{{
  "derived_question": "<the answerable question>",
  "what_changed": "<e.g. period moved from 2019 to 2021-2024; age bands coarser>",
  "serves_original": "yes | partly | no",
  "why_it_serves": "<one sentence>",
  "period_used": "<period actually available and used>",
  "dimensions_to_use": ["<dimension ids>"],
  "measure_to_use": "<measure>"
}}

If no honest question can be derived without changing the subject, set
"serves_original" to "no" and say why.
"""


def fmt_dims(rec, max_cats=8):
    out = []
    for d in (rec.get("dimensions") or [])[:8]:
        sample = ", ".join(str(x) for x in (d.get("sample") or [])[:max_cats])
        out.append(f"  - {d.get('id')} ({d.get('n_categories')} categories): {sample}")
    return "\n".join(out) or "  (none)"


def fmt_measures(rec):
    ms = rec.get("measures") or []
    return "\n".join(f"  - {m.get('name')} [{m.get('unit') or ''}]" for m in ms[:10]) \
        or "  (single value column)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive answerable, faithful questions")
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--corpus", default=CORPUS, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-n", type=int, default=2, help="datasets tried per question")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    corpus = {}
    with args.corpus.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("publisher") == "CBS":
                corpus[r["code"]] = r
    rows = [json.loads(l) for l in args.src.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    jobs = []
    for ri, r in enumerate(rows):
        # take the best candidates that are at least topically right
        cands = [c for c in (r.get("candidates") or [])
                 if (c.get("verification") or {}).get("can_answer") in ("yes", "partly")]
        for c in cands[: args.top_n]:
            rec = corpus.get(c["code"])
            if rec:
                jobs.append((ri, c["code"], r, rec))
    print(f"[INFO] {len(rows):,} questions -> {len(jobs):,} (question, dataset) derivations")

    def prompt(r, rec):
        cov = rec.get("coverage") or {}
        return USER.format(
            q=r.get("question_selfcontained") or r.get("question"),
            scope=json.dumps({k: v for k, v in (r.get("scope") or {}).items() if v},
                             ensure_ascii=False) or "(not stated)",
            title=(r.get("source_title") or r.get("source_url") or "")[:90],
            code=rec["code"], dtitle=rec.get("title_en") or rec.get("title_native"),
            coverage=(f"{cov.get('start')} to {cov.get('end')}" if cov.get("start") else "(unknown)"),
            dims=fmt_dims(rec), measures=fmt_measures(rec))

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if jobs:
            _, _, r, rec = jobs[0]
            print("\n--- sample prompt ---\n" + prompt(r, rec)[:1800])
        return
    if not jobs:
        print("[DONE] nothing to derive")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": prompt(r, rec)}] for _, _, r, rec in jobs]

    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = collections_yes = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for (ri, code, r, rec), o in zip(jobs, outs):
            obj = parse_json(o.outputs[0].text if o.outputs else "")
            if not obj or not obj.get("derived_question"):
                continue
            ok += 1
            collections_yes += int(obj.get("serves_original") == "yes")
            fh.write(json.dumps({
                "derived_question": obj["derived_question"],
                "original_question": r.get("question_selfcontained") or r.get("question"),
                "original_verbatim": r.get("question"),
                "what_changed": obj.get("what_changed"),
                "serves_original": obj.get("serves_original"),
                "why_it_serves": obj.get("why_it_serves"),
                "period_used": obj.get("period_used"),
                "dimensions_to_use": obj.get("dimensions_to_use"),
                "measure_to_use": obj.get("measure_to_use"),
                "dataset": {"code": code, "title_en": rec.get("title_en"),
                            "title_native": rec.get("title_native")},
                "publisher": "CBS",
                "source_url": r.get("source_url"),
                "source_title": r.get("source_title"),
                "answer_type": r.get("answer_type"),
            }, ensure_ascii=False) + "\n")
    print(f"[DONE] derived {ok:,} questions ({collections_yes:,} judged to fully serve the "
          f"original) -> {args.out}")


if __name__ == "__main__":
    main()
