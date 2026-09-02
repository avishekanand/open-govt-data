#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translate extracted research questions to English, in place.

Adds a `question_en` field to every research question in pub_evidence.jsonl.
The **witness sentences are deliberately left untranslated**: they are evidence,
quoted verbatim from the source, and are verified by exact match against the
document text. Translating them would destroy that guarantee.

    python -m enrich.translate_questions --dry-run
    python -m enrich.translate_questions --model Qwen/Qwen3-32B

Idempotent: questions that already carry a `question_en` are skipped.
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

SCHEMA = {
    "type": "object",
    "properties": {"english": {"type": "string"}},
    "required": ["english"],
    "additionalProperties": False,
}

SYSTEM = ("You translate research questions into natural English. Return STRICT JSON "
          "only. Preserve meaning and any domain terms; do not answer the question.")

USER = """\
Translate this research question into English. If it is already English, return it
unchanged. Keep Dutch institution names and register names (CBS, Wmo, SBI2008,
GBA/BRP, ...) as they are.

Question: {q}

Return: {{"english": "<the question in English>"}}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate research questions to English")
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.src.open(encoding="utf-8")]
    todo: List[tuple] = []          # (record_index, question_index, text)
    for ri, r in enumerate(recs):
        for qi, q in enumerate(r.get("research_questions") or []):
            if q.get("question_en"):
                continue
            if q.get("question"):
                todo.append((ri, qi, q["question"]))
    print(f"[INFO] {len(todo):,} questions to translate "
          f"(of {sum(len(r.get('research_questions') or []) for r in recs):,} total)")

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if todo:
            print("\n--- sample prompt ---\n" + USER.format(q=todo[0][2])[:500])
        return
    if not todo:
        print("[DONE] nothing to translate")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM},
              {"role": "user", "content": USER.format(q=t)}] for _, _, t in todo]

    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    ok = fail = 0
    for (ri, qi, orig), o in zip(todo, outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        en = (obj or {}).get("english")
        if isinstance(en, str) and en.strip():
            recs[ri]["research_questions"][qi]["question_en"] = en.strip()
            ok += 1
        else:
            fail += 1

    tmp = args.src.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.src)          # atomic: never leave a half-written corpus
    print(f"[DONE] translated ok={ok} fail={fail} -> {args.src}")


if __name__ == "__main__":
    main()
