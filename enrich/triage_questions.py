#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triage each question for benchmark readiness, with per-QUESTION dataset attribution.

Fixes a real defect in the first seed build: candidate datasets were attached at
DOCUMENT level, so every question in a paper inherited every dataset that paper
cited. Four kidney-disease questions ended up paired with an income-percentile
table the same paper happened to cite. Here the model must decide, for THIS
question, which of the document's candidates (if any) actually answers it - and
the returned code is validated against the candidate list, so it cannot invent one.

Adds:  benchmark_status, attributed_dataset, attribution_confidence,
       specificity, missing_to_specify, status_reason

    python -m enrich.triage_questions --dry-run
    python -m enrich.triage_questions --model Qwen/Qwen3-32B
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

SRC = Path("data/processed/pub/pub_evidence.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")

# Status is ONLY about which data could answer the question. Vagueness is a
# SEPARATE axis (`specificity`) - conflating the two collapsed 73% of questions
# into not_a_data_question in the first run, including "Which conditions lead to
# the highest healthcare expenditures?", which is a data question that merely
# needs a period pinned. `ambiguous` was removed because it duplicated
# `specificity` and the model never selected it.
STATUS = [
    "open_data",            # published CBS/Eurostat aggregate tables could answer it
    "microdata_deferred",   # needs record-level CBS data under a project licence
    "other_source",         # needs surveys/interviews/literature, not official statistics
    "not_a_data_question",  # asks WHY/mechanism, a policy judgement, or about the study's own method
]
CONFIDENCE = ["high", "medium", "low", "none"]
SPECIFICITY = ["specific", "underspecified", "vague"]

SCHEMA = {
    "type": "object",
    "properties": {
        "benchmark_status": {"type": "string", "enum": STATUS},
        "attributed_dataset": {"type": "string"},
        "attribution_confidence": {"type": "string", "enum": CONFIDENCE},
        "specificity": {"type": "string", "enum": SPECIFICITY},
        "missing_to_specify": {"type": "string"},
        "status_reason": {"type": "string"},
    },
    "required": ["benchmark_status", "attributed_dataset", "attribution_confidence",
                 "specificity", "missing_to_specify", "status_reason"],
    "additionalProperties": False,
}

SYSTEM = ("You triage research questions for a data benchmark. You are strict and "
          "conservative: attribute a dataset ONLY if it plainly answers the question. "
          "Return STRICT JSON only.")

USER = """\
Question: {q}
From publication: {title}

Datasets cited somewhere in that publication (they may be irrelevant to THIS question):
{cands}

Decide:

"attributed_dataset" - the code from the list above that would answer THIS
question. Use exactly "none" if none of them does, or if the list is empty. Do
NOT pick a dataset merely because the paper cited it; a paper on kidney disease
may also cite an income table.

"attribution_confidence" - high / medium / low / none.

"specificity" - would this question have ONE determinate answer?
  specific       - population, period and measure are clear enough
  underspecified - answerable but a period/population/measure must be pinned
  vague          - too open to have a determinate answer

"missing_to_specify" - what would have to be pinned down. Empty if specific.

"benchmark_status" - WHICH DATA could answer it. Judge only this; ignore how
vaguely it is phrased (that is "specificity", a separate field).
  open_data            published CBS StatLine / Eurostat aggregate tables could
                       answer it, even if a period or population must be pinned
  microdata_deferred   needs record-level CBS data under a project licence
  other_source         needs surveys, interviews or literature, not official statistics
  not_a_data_question  asks WHY something happens (a mechanism), makes a policy
                       judgement, or is about the study's own method or model

IMPORTANT: a question that is merely vague or missing a period/population is
STILL a data question. "Which conditions lead to the highest healthcare
expenditures?" is open_data with specificity=underspecified - NOT
not_a_data_question.

"status_reason" - one short sentence.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Triage questions for benchmark readiness")
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--min-link-score", type=float, default=0.7,
                    help="drop weak fuzzy links (e.g. 'Statistics Netherlands, 2022')")
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.src.open(encoding="utf-8")]
    todo: List[tuple] = []
    for ri, r in enumerate(recs):
        cands = []
        for x in r.get("data_mentions") or []:
            l = x["link"]
            if l["match"] in ("exact_id", "fuzzy_title") and (l.get("score") or 0) >= args.min_link_score:
                if not any(c["code"] == l["code"] for c in cands):
                    cands.append({"code": l["code"], "title": l["title"] or ""})
        title = (r.get("title") or r.get("domain") or "")[:110]
        for qi, q in enumerate(r.get("research_questions") or []):
            if q.get("benchmark_status"):
                continue
            text = q.get("question_en") or q.get("question")
            if text:
                todo.append((ri, qi, text, title, cands))
    print(f"[INFO] {len(todo):,} questions to triage; "
          f"{sum(1 for t in todo if t[4]):,} have >=1 candidate dataset "
          f"(link score >= {args.min_link_score})")

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        withc = next((t for t in todo if t[4]), todo[0] if todo else None)
        if withc:
            cands = "\n".join(f"  - {c['code']}: {c['title'][:70]}" for c in withc[4]) or "  (none)"
            print("\n--- sample prompt ---\n"
                  + USER.format(q=withc[2], title=withc[3], cands=cands)[:900])
        return
    if not todo:
        print("[DONE] nothing to triage")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = []
    for _, _, text, title, cands in todo:
        clist = "\n".join(f"  - {c['code']}: {c['title'][:70]}" for c in cands) or "  (none)"
        convs.append([{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": USER.format(q=text, title=title, cands=clist)}])

    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = fail = invented = 0
    for (ri, qi, _, _, cands), o in zip(todo, outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        if not obj or "benchmark_status" not in obj:
            fail += 1
            continue
        # The model may only choose from this question's candidates.
        code = (obj.get("attributed_dataset") or "none").strip()
        valid = {c["code"] for c in cands}
        if code not in valid and code.lower() not in ("none", ""):
            invented += 1
            code = "none"
            obj["attribution_confidence"] = "none"

        obj["attributed_dataset"] = code if code.lower() != "none" else None
        # gold-readiness is DERIVED, never taken from the model: open data, a
        # dataset attributed to this question, and specific enough to score.
        obj["gold_ready"] = bool(obj.get("benchmark_status") == "open_data"
                                 and obj["attributed_dataset"]
                                 and obj.get("specificity") == "specific")
        recs[ri]["research_questions"][qi].update(
            {k: obj[k] for k in ("benchmark_status", "attributed_dataset",
                                 "attribution_confidence", "specificity",
                                 "missing_to_specify", "status_reason",
                                 "gold_ready") if k in obj})
        ok += 1

    tmp = args.src.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.src)
    print(f"[DONE] triaged ok={ok} fail={fail} rejected_invented_codes={invented} -> {args.src}")


if __name__ == "__main__":
    main()
