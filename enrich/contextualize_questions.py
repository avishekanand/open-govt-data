#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rewrite each extracted question into a self-contained, answerable form.

Questions were extracted verbatim, which strips them of the context that made
them meaningful. "Which conditions lead to the highest healthcare expenditures?"
looks hopeless alone; inside its report it means Dutch healthcare spending, in a
stated year, for a stated population. That lost context is why an earlier triage
pass judged 94.7% of questions "vague" - the vagueness was an artefact of our own
extraction, not of the research.

This pass puts the context back, using the document window around the question's
own witness sentence. It is decontextualisation in the sense used for
conversational QA: make the question standalone without changing what it asks.

Adds: question_selfcontained, scope{period, geography, population, measure},
      context_found

    python -m enrich.contextualize_questions --dry-run
    python -m enrich.contextualize_questions --model Qwen/Qwen3-32B

The verbatim original is never overwritten - provenance depends on it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends, llm_kwargs

SRC = Path("data/processed/pub/pub_evidence.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
WINDOW = 2500          # characters of document context around the witness

SCHEMA = {
    "type": "object",
    "properties": {
        "question_selfcontained": {"type": "string"},
        "period": {"type": "string"},
        "geography": {"type": "string"},
        "population": {"type": "string"},
        "measure": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["question_selfcontained", "period", "geography", "population",
                 "measure", "confidence"],
    "additionalProperties": False,
}

SYSTEM = ("You make research questions self-contained using the surrounding text of the "
          "publication they came from. You never change what is being asked and never "
          "invent details that the context does not support. Return STRICT JSON only.")

USER = """\
Publication: {title}

Context from the publication (the question was asked in this passage):
\"\"\"
{context}
\"\"\"

Question as written: {q}

Rewrite it so it stands alone, using ONLY what the context supports. Fill in the
subject, the time period, the geography and the population where the context
states them - a reader who has not seen the publication should be able to act on
the rewritten question.

Rules:
- Do NOT change what is being asked, and do NOT answer it.
- Do NOT invent a period, region or population that the context does not state.
  Leave a field as "" if the context does not say.
- Keep it one sentence where possible.

Return JSON:
{{
  "question_selfcontained": "<standalone version>",
  "period": "<e.g. 2017-2019, or \\"\\" if not stated>",
  "geography": "<e.g. Netherlands, Amsterdam, EU27, or \\"\\">",
  "population": "<e.g. households, persons 65+, SMEs, or \\"\\">",
  "measure": "<what is being counted or measured, or \\"\\">",
  "confidence": "high | medium | low"
}}
"""

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip().lower()


def context_window(text: str, witness: str, window: int = WINDOW) -> Optional[str]:
    """The passage around the witness sentence; None if the witness cannot be located."""
    if not text:
        return None
    hay, needle = _norm(text), _norm(witness)
    if not needle:
        return None
    pos = hay.find(needle[: max(30, int(len(needle) * 0.6))])
    if pos < 0:
        return None
    # map back approximately: normalisation only collapses whitespace
    ratio = len(text) / max(len(hay), 1)
    centre = int(pos * ratio)
    half = window // 2
    return text[max(0, centre - half): centre + half]


def doc_text(row: Dict[str, Any]) -> str:
    tp = row.get("text_path")
    if tp and Path(tp).exists():
        try:
            return Path(tp).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Make questions self-contained from context")
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--only", default=None,
                    help="restrict to a data_needed value, e.g. public_aggregate")
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.src.open(encoding="utf-8")]
    todo: List[tuple] = []
    no_ctx = 0
    for ri, r in enumerate(recs):
        qs = r.get("research_questions") or []
        if not qs:
            continue
        text = doc_text(r) if any(not q.get("question_selfcontained") for q in qs) else ""
        title = (r.get("title") or r.get("domain") or "")[:120]
        for qi, q in enumerate(qs):
            if q.get("question_selfcontained"):
                continue
            if args.only and q.get("data_needed") != args.only:
                continue
            ctx = context_window(text, q.get("witness") or "", args.window)
            if ctx is None:
                no_ctx += 1
                ctx = (q.get("witness") or "")[:args.window]   # fall back to the witness alone
            todo.append((ri, qi, q.get("question_en") or q.get("question"), title, ctx))
    print(f"[INFO] {len(todo):,} questions to contextualise "
          f"({no_ctx:,} had no locatable window; witness used instead)")

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if todo:
            ri, qi, qt, ti, ctx = todo[0]
            print("\n--- sample prompt ---\n"
                  + USER.format(title=ti, context=ctx[:900], q=qt)[:1500])
        return
    if not todo:
        print("[DONE] nothing to do")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(title=ti, context=ctx, q=qt)}]
             for _, _, qt, ti, ctx in todo]

    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = fail = 0
    for (ri, qi, _, _, ctx), o in zip(todo, outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        if obj and obj.get("question_selfcontained"):
            q = recs[ri]["research_questions"][qi]
            q["question_selfcontained"] = obj["question_selfcontained"].strip()
            q["scope"] = {k: (obj.get(k) or "").strip()
                          for k in ("period", "geography", "population", "measure")}
            q["context_confidence"] = obj.get("confidence")
            ok += 1
        else:
            fail += 1

    tmp = args.src.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.src)
    print(f"[DONE] contextualised ok={ok} fail={fail} -> {args.src}")


if __name__ == "__main__":
    main()
