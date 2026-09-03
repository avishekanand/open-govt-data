#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that a retrieved dataset can actually answer the question.

Retrieval gives topical similarity. That is not enough for gold: a dataset can be
about the right subject and still lack the period, the breakdown, or the measure
the question needs. This pass checks each candidate against the dataset's REAL
metadata - coverage years, dimension names with their actual category values, and
measures - and asks whether the question is answerable from it, and how.

A pair is only gold if the model can name the dimensions and measure it would
use. That makes the outcome something that can genuinely be *retrieved and then
answered*, which is the point.

Adds per candidate: can_answer, dimensions_to_use, measure_to_use, missing, why.

    python -m enrich.verify_candidates --dry-run
    python -m enrich.verify_candidates --top-n 3
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

CANDS_TMPL = "data/processed/benchmark/question_dataset_candidates_{pub}.jsonl"
CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
OUT_TMPL = "data/processed/benchmark/question_dataset_verified_{pub}.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")

SCHEMA = {
    "type": "object",
    "properties": {
        "can_answer": {"type": "string", "enum": ["yes", "partly", "no"]},
        "dimensions_to_use": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "measure_to_use": {"type": "string"},
        "missing": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["can_answer", "dimensions_to_use", "measure_to_use", "missing", "why"],
    "additionalProperties": False,
}

SYSTEM = ("You judge whether a specific statistical table can answer a specific question. "
          "You are given the table's real dimensions, their actual category values, its "
          "coverage and its measures. Be concrete and strict. Return STRICT JSON only.")

USER = """\
Question: {q}
Requested scope: {scope}

Candidate dataset {code} — {title}
Publisher: {publisher}
Coverage: {coverage}
Observations: {nobs}

Dimensions (with real category values):
{dims}

Measures:
{measures}

Can this table answer the question?
  yes    - the subject, the needed breakdown and the measure are present
  partly - the subject matches but a breakdown or the measure is missing
  no     - it cannot answer it at all

PERIOD RULE - important. Official tables are living: coverage shifts, and a
question taken from a 2019 report often names a year the current table no longer
carries. Judge the SUBJECT, the BREAKDOWN and the MEASURE strictly, but treat the
period as a PREFERENCE, not a requirement:
  * if the table has a time dimension and the measure and breakdown fit, answer
    "yes" even when the requested year is absent, and say in "missing" which
    period IS available (the question will be re-scoped to it);
  * answer "no" on period grounds ONLY if the table has no usable time dimension.

"dimensions_to_use" - the dimension ids you would filter or group by (exact ids
from the list above; empty if "no").
"measure_to_use"    - the measure that carries the answer ("" if "no").
"missing"           - what is absent, if anything. If the requested period is
                      unavailable, state the period the table DOES cover.
"why"               - one short sentence.
"""


def fmt_dims(rec: Dict[str, Any], max_cats: int = 8) -> str:
    out = []
    for d in (rec.get("dimensions") or [])[:8]:
        sample = ", ".join(str(x) for x in (d.get("sample") or [])[:max_cats])
        out.append(f"  - {d.get('id')} ({d.get('n_categories')} categories): {sample}")
    return "\n".join(out) or "  (none recorded)"


def fmt_measures(rec: Dict[str, Any]) -> str:
    ms = rec.get("measures") or []
    if ms:
        return "\n".join(f"  - {m.get('name')} [{m.get('unit') or ''}]" for m in ms[:10])
    return "  (single value column; unit given by the 'unit' dimension)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify retrieved datasets can answer the question")
    ap.add_argument("--publisher", default="CBS", choices=["CBS", "ESTAT"],
                    help="which benchmark to verify; they are kept separate")
    ap.add_argument("--cands", default=None, type=Path)
    ap.add_argument("--corpus", default=CORPUS, type=Path)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--top-n", type=int, default=3, help="candidates verified per question")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.cands = args.cands or Path(CANDS_TMPL.format(pub=args.publisher.lower()))
    args.out = args.out or Path(OUT_TMPL.format(pub=args.publisher.lower()))
    corpus = {json.loads(l)["code"]: json.loads(l) for l in args.corpus.open(encoding="utf-8")
              if json.loads(l).get("publisher") == args.publisher}
    rows = [json.loads(l) for l in args.cands.open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    jobs: List[tuple] = []
    for ri, r in enumerate(rows):
        for ci, c in enumerate((r.get("candidates") or [])[: args.top_n]):
            rec = corpus.get(c["code"])
            if rec:
                jobs.append((ri, ci, r, rec))
    print(f"[INFO] {len(rows):,} questions x top-{args.top_n} = {len(jobs):,} pairs to verify")

    def prompt(r, rec):
        cov = rec.get("coverage") or {}
        return USER.format(
            q=r.get("question_selfcontained") or r.get("question"),
            scope=json.dumps({k: v for k, v in (r.get("scope") or {}).items() if v},
                             ensure_ascii=False) or "(not stated)",
            code=rec.get("code"), title=rec.get("title_en") or rec.get("title_native"),
            publisher=rec.get("publisher"),
            coverage=(f"{cov.get('start')} to {cov.get('end')}"
                      if cov.get("start") else "(unknown)"),
            nobs=rec.get("n_observations"), dims=fmt_dims(rec), measures=fmt_measures(rec))

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if jobs:
            _, _, r, rec = jobs[0]
            print("\n--- sample prompt ---\n" + prompt(r, rec)[:1500])
        return
    if not jobs:
        print("[DONE] nothing to verify")
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

    ok = 0
    for (ri, ci, _, _), o in zip(jobs, outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        if obj and obj.get("can_answer"):
            rows[ri]["candidates"][ci]["verification"] = {
                k: obj.get(k) for k in ("can_answer", "dimensions_to_use",
                                        "measure_to_use", "missing", "why")}
            ok += 1

    # A question is answerable if any verified candidate says yes.
    n_yes = 0
    for r in rows:
        best = None
        for c in r.get("candidates") or []:
            v = c.get("verification") or {}
            if v.get("can_answer") == "yes":
                best = c
                break
        r["verified_dataset"] = best["code"] if best else None
        r["verified_how"] = (best.get("verification") if best else None)
        cited = (r.get("lexically_attributed") or "").upper()
        r["agrees_with_citation"] = bool(best and cited and best["code"].upper() == cited)
        n_yes += int(bool(best))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    agree = sum(1 for r in rows if r.get("agrees_with_citation"))
    withcite = sum(1 for r in rows if r.get("lexically_attributed"))
    print(f"[DONE] verified {ok}/{len(jobs)} pairs -> {args.out}")
    print(f"[STAT] questions with an answerable dataset: {n_yes:,}/{len(rows):,}")
    print(f"[STAT] agreement with the paper's own citation: {agree}/{withcite}")


if __name__ == "__main__":
    main()
