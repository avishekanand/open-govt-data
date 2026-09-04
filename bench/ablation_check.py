#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation: can the question be answered with only ONE of its two tables?

The generator was asked to self-report `single_table_answerable` and returned
false 109 times out of 109. A model that has just done the work is not a
credible judge of whether the work was necessary.

So this asks independently. Each question is shown with ONE table's real schema
and no hint that a second exists, and asked whether it can be answered. If
either single-table view says yes, the item is DISCONNECTED - it looks two-hop
but is not - and it is dropped. This is MuSiQue's ablation, and it is the
standard way disconnected reasoning is caught.

    python -m bench.ablation_check --tier 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends, llm_kwargs

PROFILES = Path("data/processed/field_profiles_estat.jsonl")
IN_TMPL = "data/processed/benchmark/hop_questions_tier{tier}.jsonl"
OUT_TMPL = "data/processed/benchmark/hop_questions_tier{tier}_checked.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B-AWQ")

SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "string", "enum": ["yes", "partly", "no"]},
        "what_is_missing": {"type": "string", "maxLength": 160},
        "why": {"type": "string", "maxLength": 160},
    },
    "required": ["answerable", "what_is_missing", "why"],
    "additionalProperties": False,
}

SYSTEM = ("You judge whether ONE statistical table is sufficient to answer a question. "
          "You see only this table. Be strict and literal about what it contains. "
          "Return STRICT JSON only.")

USER = """\
Question: {q}

The only table available:
  {code} — {title}
  Coverage: {cov}
  Fields:
{fields}

Could this question be answered completely from THIS table alone?
  yes    - every quantity the question needs is in this table
  partly - some of it is here, but at least one needed quantity is absent
  no     - this table cannot answer it

"what_is_missing" - the quantity that is not in this table, if any.
"""


def fmt_fields(prof, max_f=7, max_c=8):
    out = []
    for name, f in list((prof.get("fields") or {}).items())[:max_f]:
        s = ", ".join(str(v) for v in (f.get("sample") or [])[:max_c])
        out.append(f"    - {name} ({f.get('n_categories')}): {s}")
    ms = prof.get("measures") or []
    if ms:
        out.append("    measures: " + ", ".join(str(m.get("name")) for m in ms[:8]))
    return "\n".join(out) or "    (none)"


def cov(prof):
    p = prof.get("period") or {}
    return (f"{p.get('first_year')}–{p.get('last_year')}" if p.get("first_year") else "(unknown)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-table ablation for two-hop questions")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--limit-questions", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prof = {json.loads(l)["code"].lower(): json.loads(l) for l in PROFILES.open(encoding="utf-8")}
    rows = [json.loads(l) for l in Path(IN_TMPL.format(tier=args.tier)).open(encoding="utf-8")]
    if args.limit_questions:
        rows = rows[: args.limit_questions]
    jobs = []
    for ri, r in enumerate(rows):
        for side, d in enumerate(r["datasets"]):
            pr = prof.get(d["code"].lower())
            if pr:
                jobs.append((ri, side, r, d, pr))
    print(f"[INFO] {len(rows):,} questions x 2 tables = {len(jobs):,} ablations")

    def prompt(r, d, pr):
        return USER.format(q=r["question"], code=d["code"],
                           title=d.get("title_en") or d["code"],
                           cov=cov(pr), fields=fmt_fields(pr))

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if jobs:
            _, _, r, d, pr = jobs[0]
            print("\n--- sample prompt ---\n" + prompt(r, d, pr)[:1200])
        return
    if not jobs:
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": prompt(r, d, pr)}] for _, _, r, d, pr in jobs]
    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    n_empty = n_unparsed = 0
    for (ri, side, r, d, _), o in zip(jobs, outs):
        raw = o.outputs[0].text if o.outputs else ""
        obj = parse_json(raw) or {}
        if not raw.strip():
            n_empty += 1
        elif not obj:
            n_unparsed += 1
        entry = {"answerable": obj.get("answerable"), "missing": obj.get("what_is_missing"),
                 "why": obj.get("why")}
        if not obj:
            # keep the evidence: a silent None is how 135 unchecked views were
            # certified as fine in the first run
            entry["raw"] = raw[:300]
            entry["finish_reason"] = getattr(o.outputs[0], "finish_reason", None) if o.outputs else None
        rows[ri].setdefault("ablation", {})[d["code"]] = entry
    print(f"[DIAG] empty generations {n_empty} | unparsed non-empty {n_unparsed} "
          f"of {len(jobs)}")

    kept = dropped = unchecked = 0
    out = Path(OUT_TMPL.format(tier=args.tier))
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            abl = r.get("ablation") or {}
            verdicts = [v.get("answerable") for v in abl.values()]
            # An item is connected only if BOTH tables were actually judged and
            # NEITHER sufficed. A missing verdict means the check did not run -
            # counting that as a pass silently certified 135 of 218 unchecked
            # views as fine, which is how the first run reported 109/109.
            r["connected"] = (len(verdicts) == 2
                              and all(v in ("no", "partly") for v in verdicts))
            r["unchecked"] = any(v is None for v in verdicts) or len(verdicts) < 2
            r["ablation_verdicts"] = verdicts
            if r.get("unchecked"):
                unchecked += 1
            elif r["connected"]:
                kept += 1
            else:
                dropped += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] connected {kept:,} | disconnected (one table sufficed) {dropped:,} "
          f"| UNCHECKED (no verdict) {unchecked:,} -> {out}")


if __name__ == "__main__":
    main()
