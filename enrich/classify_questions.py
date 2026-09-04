#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classify extracted research questions by what kind of answer they admit.

Most of the 3,276 attested questions are not (query, single answer) pairs: some
ask for a trend or a plot, some for a mechanism, some are about the study's own
methodology, many need confidential microdata. Before any of them can seed a
benchmark, we need to know which subset is actually *verifiable now* - i.e.
answerable by querying public aggregate tables to produce a concrete value.

Adds to each question:  answer_type, answer_shape, data_needed, publisher_hint,
verifiable_now, reason.

    python -m enrich.classify_questions --dry-run
    python -m enrich.classify_questions --model Qwen/Qwen3-32B

Idempotent (skips questions already classified) and atomic (temp file + replace).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends, llm_kwargs

SRC = Path("data/processed/pub/pub_evidence.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")

ANSWER_TYPES = [
    "single_number",        # a count or amount: "how many X in year Y"
    "rate_or_share",        # a percentage or proportion
    "comparison",           # difference/ratio between groups or periods
    "trend",                # change over time - a series, often a plot
    "ranking_or_list",      # which regions/sectors, top-k
    "distribution",         # breakdown across categories - a table or plot
    "yes_no",               # a factual yes/no answerable from data
    "qualitative",          # mechanism, cause, explanation - not a lookup
    "methodological",       # about the study's own method or data quality
    "normative",            # what should be done - policy recommendation
    "other",
]
DATA_NEEDED = ["public_aggregate", "microdata", "other_source", "none"]
PUBLISHERS = ["CBS", "ESTAT", "either", "not_applicable"]

SCHEMA = {
    "type": "object",
    "properties": {
        "answer_type": {"type": "string", "enum": ANSWER_TYPES},
        "data_needed": {"type": "string", "enum": DATA_NEEDED},
        "publisher_hint": {"type": "string", "enum": PUBLISHERS},
        "verifiable_now": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["answer_type", "data_needed", "publisher_hint", "verifiable_now", "reason"],
    "additionalProperties": False,
}

SYSTEM = ("You classify research questions by what kind of answer they admit and what "
          "data would be needed. You are strict: most questions are NOT answerable as a "
          "single number. Return STRICT JSON only.")

USER = """\
Research question (from the publication "{title}"):
{q}

Classify it.

"answer_type" - the shape of a correct answer:
  single_number, rate_or_share, comparison, trend, ranking_or_list, distribution,
  yes_no, qualitative (a mechanism or explanation), methodological (about the
  study's own method or data quality), normative (what policy should do), other.

"data_needed":
  public_aggregate - answerable from published CBS StatLine / Eurostat tables
  microdata        - needs record-level CBS data under a project licence
  other_source     - needs a survey, interviews, admin data or literature that is
                     not CBS/Eurostat statistics
  none             - not an empirical data question at all

"publisher_hint" - whose statistics would answer it: CBS (Netherlands),
ESTAT (Eurostat, cross-country/EU), either, or not_applicable.

"verifiable_now" - true ONLY if a concrete answer could be produced today by
querying published aggregate tables. A question needing microdata, interviews,
causal inference, or a value judgement is false.

"reason" - one short sentence.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify research questions by answerability")
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.src.open(encoding="utf-8")]
    todo: List[tuple] = []
    for ri, r in enumerate(recs):
        title = (r.get("title") or r.get("domain") or "")[:120]
        for qi, q in enumerate(r.get("research_questions") or []):
            if q.get("answer_type"):
                continue
            text = q.get("question_en") or q.get("question")
            if text:
                todo.append((ri, qi, text, title))
    total = sum(len(r.get("research_questions") or []) for r in recs)
    print(f"[INFO] {len(todo):,} questions to classify (of {total:,})")

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if todo:
            print("\n--- sample prompt ---\n"
                  + USER.format(q=todo[0][2], title=todo[0][3])[:700])
        return
    if not todo:
        print("[DONE] nothing to classify")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(q=t, title=ti)}] for _, _, t, ti in todo]

    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = fail = 0
    for (ri, qi, _, _), o in zip(todo, outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        if obj and "answer_type" in obj:
            recs[ri]["research_questions"][qi].update(
                {k: obj[k] for k in ("answer_type", "data_needed", "publisher_hint",
                                     "verifiable_now", "reason") if k in obj})
            ok += 1
        else:
            fail += 1

    tmp = args.src.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.src)
    print(f"[DONE] classified ok={ok} fail={fail} -> {args.src}")


if __name__ == "__main__":
    main()
