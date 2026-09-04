#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite questions that merely restate their table's title.

27% of the constructed Eurostat questions had >0.5 token overlap with the title
of their own gold dataset - "What was the circular material use rate in the EU in
2024?" against a table called "Circular Material Use Rate". Those are solved by
string matching, so as *discovery* items they measure nothing.

This rewrites them in a plain or conversational register using the surface-form
ladder, with the table's distinctive title words (`avoid_words`) forbidden. The
question must keep exactly the same meaning and the same answer - only the
wording changes.

Overlap is re-measured afterwards by scripts/check_leakage.py against the 0.18
baseline set by attested questions, which were written by researchers who had
never seen this catalogue.

    python -m enrich.deleak_questions --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends, llm_kwargs

CONSTRUCTED = Path("data/processed/benchmark/constructed_se.jsonl")
SURFACE = Path("data/processed/surface_forms_estat.jsonl")
OUT = Path("data/processed/benchmark/constructed_se_deleaked.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B-AWQ")

WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "on", "by", "in", "for", "and", "to", "from", "with", "per",
        "what", "which", "how", "many", "much", "is", "are", "was", "were", "did", "do",
        "does", "number", "total", "eu", "european", "union", "netherlands", "dutch", "average"}


def toks(s):
    return {w for w in WORD.findall((s or "").lower()) if w not in STOP}


def overlap(q, title):
    a, b = toks(q), toks(title)
    return len(a & b) / len(b) if b else 0.0


SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "register": {"type": "string",
                     "enum": ["plain", "conversational", "action_oriented", "idiomatic"]},
        "meaning_preserved": {"type": "boolean"},
        "note": {"type": "string"},
    },
    "required": ["question", "register", "meaning_preserved", "note"],
    "additionalProperties": False,
}

SYSTEM = ("You rephrase a data question in everyday English without changing what it asks "
          "or what its answer would be. Return STRICT JSON only.")

USER = """\
Question (currently phrased in the same technical words as the table it comes from):
  "{q}"

The quantity it asks about, said in other ways:
{ladder}

Words you must NOT use (they are the table's own title words):
  {avoid}

Rewrite the question so that:
  * it asks EXACTLY the same thing, with exactly the same answer;
  * it keeps the period, the place and any breakdown named in the original;
  * it avoids every forbidden word above, including plurals and variants;
  * it sounds like a person asking, not a table caption.

Critically: preserve the nature of the quantity. If it is a rate adjusted for
comparison, the rewrite must still convey an adjusted or comparable rate - never
turn it into a plain count. If you cannot rewrite it without changing the
meaning, set "meaning_preserved" to false and explain in "note".

Return JSON:
{{"question": "...", "register": "plain | conversational | action_oriented | idiomatic",
  "meaning_preserved": true/false, "note": "..."}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="De-leak constructed questions")
    ap.add_argument("--constructed", default=CONSTRUCTED, type=Path)
    ap.add_argument("--surface", default=SURFACE, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.3,
                    help="rewrite questions whose title overlap exceeds this")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=3072)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    surf = {}
    with args.surface.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            surf[r["code"].lower()] = r
    rows = [json.loads(l) for l in args.constructed.open(encoding="utf-8")]

    jobs, passthrough = [], []
    for r in rows:
        title = r["gold_dataset"].get("title_en") or ""
        o = overlap(r["question"], title)
        r["title_overlap"] = round(o, 3)
        if o > args.threshold and r["gold_dataset"]["code"].lower() in surf:
            jobs.append(r)
        else:
            passthrough.append(r)
    print(f"[INFO] {len(rows):,} constructed | {len(jobs):,} above overlap {args.threshold} "
          f"to rewrite | {len(passthrough):,} kept as-is")

    def ladder_text(code):
        s = surf.get(code.lower(), {})
        out = []
        for k in ("plain", "conversational", "action_oriented", "idiomatic"):
            for v in (s.get(k) or [])[:2]:
                out.append(f"  {k}: {v}")
        return "\n".join(out) or "  (none available)"

    def prompt(r):
        s = surf.get(r["gold_dataset"]["code"].lower(), {})
        return USER.format(q=r["question"], ladder=ladder_text(r["gold_dataset"]["code"]),
                           avoid=", ".join(s.get("avoid_words") or []) or "(none)")

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
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(r)}]
             for r in jobs]
    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    rewritten = failed = refused = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in passthrough:
            r["deleaked"] = False
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r, o in zip(jobs, outs):
            obj = parse_json(o.outputs[0].text if o.outputs else "")
            if not obj or not obj.get("question"):
                failed += 1
                r["deleaked"] = False
            elif not obj.get("meaning_preserved"):
                refused += 1
                r["deleaked"] = False
                r["deleak_note"] = obj.get("note")
            else:
                new_o = overlap(obj["question"], r["gold_dataset"].get("title_en") or "")
                r["question_original_phrasing"] = r["question"]
                r["question"] = obj["question"]
                r["register"] = obj.get("register")
                r["title_overlap_before"] = r["title_overlap"]
                r["title_overlap"] = round(new_o, 3)
                r["deleaked"] = True
                rewritten += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[DONE] rewritten {rewritten:,} | refused (meaning would change) {refused:,} | "
          f"failed {failed:,} -> {args.out}")


if __name__ == "__main__":
    main()
