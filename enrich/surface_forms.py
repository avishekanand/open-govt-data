#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A register ladder for the jargon in each table: how people actually say it.

Statistical tables are named in a technical register - "standardised incidence
rate of fatal accidents at work". Real users do not ask in that register, and a
question that reuses it is answerable by string matching, which is why 27% of the
constructed Eurostat questions failed the leakage check.

For each table's distinctive terms this produces the same idea said several ways:

  formal        standardised incidence rate of fatal accidents at work
  plain         comparable rate of deadly workplace accidents
  conversational  how high was the death rate at work
  idiomatic     an apples-to-apples rate of workplace deaths
  action        the fatality rate, scaled to account for workforce differences

Two uses:
  * de-leaking  - phrase a question without the table's own title words
  * robustness  - the same question in five registers is an evaluation axis:
                  does retrieval survive plain English?

    python -m enrich.surface_forms --publisher ESTAT --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

PROFILES = "data/processed/field_profiles_{pub}.jsonl"
CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
OUT_TMPL = "data/processed/surface_forms_{pub}.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B-AWQ")

REGISTERS = ["formal", "plain", "conversational", "idiomatic", "action_oriented"]

SCHEMA = {
    "type": "object",
    "properties": {
        "concept": {"type": "string"},
        "formal": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
        "plain": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 4},
        "conversational": {"type": "array", "items": {"type": "string"},
                           "minItems": 1, "maxItems": 3},
        "idiomatic": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "action_oriented": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "avoid_words": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
    "required": ["concept", "formal", "plain", "conversational", "idiomatic",
                 "action_oriented", "avoid_words"],
    "additionalProperties": False,
}

SYSTEM = ("You explain statistical jargon the way different people actually say it. "
          "You never change the meaning - a rate stays a rate, a count stays a count. "
          "Return STRICT JSON only.")

USER = """\
A statistical table is described as:
  {title}
Its measured quantities / units: {measures}
Key breakdowns: {dims}

Take the ONE central quantity this table reports (the "concept") and give the
ways a person might refer to it, without changing what it means.

  "formal"          - the technical phrasing a statistician would use
  "plain"           - plain English a non-specialist would understand; say what
                      the adjustment or normalisation actually does
  "conversational"  - how someone would say it out loud in a question
  "idiomatic"       - natural English idioms if any genuinely fit (e.g. an
                      apples-to-apples rate, a level-playing-field comparison);
                      return [] if none fits - do NOT force one
  "action_oriented" - replace the dense noun phrase with a descriptive action
                      (e.g. "the fatality rate, scaled for workforce size")
  "avoid_words"     - the distinctive words from the TITLE that a paraphrase
                      should avoid if it wants to not simply restate the title

Meaning is preserved exactly. A "standardised" rate must still convey that it is
adjusted for comparison; do not silently turn it into a raw count.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Register ladder per table")
    ap.add_argument("--publisher", choices=["CBS", "ESTAT"], required=True)
    ap.add_argument("--out", default=None, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only-cited", action="store_true",
                    help="restrict to datasets cited by an article (the benchmark pool)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out = args.out or Path(OUT_TMPL.format(pub=args.publisher.lower()))

    keep = None
    if args.only_cited:
        keep = set()
        se = Path("data/processed/estat/se_articles.jsonl")
        if args.publisher == "ESTAT" and se.exists():
            with se.open(encoding="utf-8") as fh:
                for line in fh:
                    keep |= {c.lower() for c in (json.loads(line).get("codes_in_catalogue") or [])}
        cons = Path("data/processed/benchmark/constructed_se.jsonl")
        if cons.exists():
            with cons.open(encoding="utf-8") as fh:
                for line in fh:
                    keep.add(json.loads(line)["gold_dataset"]["code"].lower())

    profiles = {}
    with open(PROFILES.format(pub=args.publisher.lower()), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            profiles[r["code"]] = r

    jobs = []
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("publisher") != args.publisher:
                continue
            if keep is not None and r["code"].lower() not in keep:
                continue
            prof = profiles.get(r["code"], {})
            units = []
            for f in (prof.get("fields") or {}).values():
                pass
            measures = [m.get("name") for m in (r.get("measures") or [])][:8]
            unit_field = (prof.get("fields") or {}).get("unit") or {}
            measures += unit_field.get("sample", [])[:6]
            dims = [d for d in (prof.get("fields") or {}) if d not in ("time", "freq")][:6]
            jobs.append({"code": r["code"], "title": r.get("title_en") or r.get("title_native"),
                         "measures": ", ".join(str(m) for m in measures if m) or "(a single value column)",
                         "dims": ", ".join(dims) or "(none)"})
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[INFO] {len(jobs):,} {args.publisher} tables to build surface forms for")

    def prompt(j):
        return USER.format(title=j["title"], measures=j["measures"], dims=j["dims"])

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if jobs:
            print("\n--- sample prompt ---\n" + prompt(jobs[0])[:1400])
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
            if not obj or not obj.get("concept"):
                continue
            ok += 1
            fh.write(json.dumps({"code": j["code"], "title_en": j["title"],
                                 "publisher": args.publisher, **obj}, ensure_ascii=False) + "\n")
    print(f"[DONE] {ok:,} surface-form entries -> {args.out}")


if __name__ == "__main__":
    main()
