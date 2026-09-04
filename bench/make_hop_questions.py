#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write two-hop questions over grounded dataset pairs.

Each question must genuinely need BOTH tables. The standard failure of
templated multi-hop construction is a question that reads as two-hop but is
answerable from one table - MuSiQue's "disconnected reasoning". So the model
must state what each hop contributes and why neither table alone suffices, and
those claims are checkable by a reviewer against the two schemas.

Grounding comes from bench.hop_pairs tiers:
  1  the article relates the two tables in one prose passage  (strongest)
  2  the article cites both, elsewhere
  3  the domain crossing is attested elsewhere in the corpus

    python -m bench.make_hop_questions --tier 1 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends, llm_kwargs

PAIRS = Path("data/processed/benchmark/hop_pairs.jsonl")
PROFILES = Path("data/processed/field_profiles_estat.jsonl")
SURFACE = Path("data/processed/surface_forms_estat.jsonl")
OUT_TMPL = "data/processed/benchmark/hop_questions_tier{tier}.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B-AWQ")

SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "hop_1": {"type": "string"},
        "hop_2": {"type": "string"},
        "why_both_needed": {"type": "string"},
        "join_on": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "period_used": {"type": "string"},
        "economic_concept": {"type": "string"},
        "answer_shape": {"type": "string",
                         "enum": ["single_value", "series_over_time",
                                  "series_over_countries", "comparison_of_two"]},
        "single_table_answerable": {"type": "boolean"},
    },
    "required": ["question", "hop_1", "hop_2", "why_both_needed", "join_on",
                 "period_used", "economic_concept", "answer_shape",
                 "single_table_answerable"],
    "additionalProperties": False,
}

SYSTEM = ("You write analytical questions that require combining two statistical tables. "
          "You never write a question that one table alone could answer. "
          "Return STRICT JSON only.")

USER = """\
Two tables that {evidence}{article_line}

TABLE A — {a} — {title_a}   [{domain_a}]
  Coverage: {cov_a}
  Fields:
{fields_a}

TABLE B — {b} — {title_b}   [{domain_b}]
  Coverage: {cov_b}
  Fields:
{fields_b}

They can be joined on: {join_keys}
{passage_block}
Write ONE question that a researcher would actually ask, which:
  1. needs a value from TABLE A **and** a value from TABLE B - a ratio, a
     comparison, a per-capita or per-GDP normalisation, a share, a correlation
     across countries or years;
  2. names a period both tables cover, and categories that exist. The answer does
     NOT have to be one number - a question whose answer is a SERIES is often the
     more natural one to ask, e.g. "how did the value per tonne of potatoes move
     between 2021 and 2024" (a line per year), or "how does the organic share
     differ across member states in 2024" (a value per country). Prefer a series
     when that is how the relationship would actually be examined;
  3. is about a real economic or social relationship - not a puzzle. Avoid
     constructions like "what is X in the country with the highest Y" unless
     that is genuinely how the relationship is studied;
  4. avoids restating either table's title wording.

Return JSON:
{{
  "question": "...",
  "hop_1": "<what is taken from TABLE A>",
  "hop_2": "<what is taken from TABLE B>",
  "why_both_needed": "<why one table alone cannot answer it>",
  "join_on": ["<the dimensions the two are matched on>"],
  "period_used": "<a period both cover>",
  "economic_concept": "<the relationship being examined, in a few words>",
  "answer_shape": "single_value | series_over_time | series_over_countries | comparison_of_two",
  "single_table_answerable": false
}}

Set "single_table_answerable" to true if you could not avoid writing a question
that one table alone answers - honesty here matters more than producing an item.
"""


def fmt_fields(prof: Dict[str, Any], max_f: int = 6, max_c: int = 8) -> str:
    out = []
    for name, f in list((prof.get("fields") or {}).items())[:max_f]:
        s = ", ".join(str(v) for v in (f.get("sample") or [])[:max_c])
        out.append(f"    - {name} ({f.get('n_categories')}): {s}")
    ms = prof.get("measures") or []
    if ms:
        out.append("    measures: " + ", ".join(str(m.get("name")) for m in ms[:6]))
    return "\n".join(out) or "    (none)"


def cov(prof: Dict[str, Any]) -> str:
    p = prof.get("period") or {}
    return f"{p.get('first_year')}–{p.get('last_year')} ({p.get('granularity','annual')})" \
        if p.get("first_year") else "(unknown)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-hop question construction")
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prof = {json.loads(l)["code"].lower(): json.loads(l) for l in PROFILES.open(encoding="utf-8")}
    pairs = [json.loads(l) for l in PAIRS.open(encoding="utf-8") if json.loads(l)["tier"] == args.tier]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[INFO] tier {args.tier}: {len(pairs):,} grounded pairs")

    def prompt(p):
        pa, pb = prof[p["a"].lower()], prof[p["b"].lower()]
        pb_block = (f'\nThe article relates them here:\n"""\n{p["passage"][:900]}\n"""\n'
                    if p.get("passage") else "\n")
        return USER.format(
            evidence=p["evidence"],
            article_line=(f', in "{p["article"]}"' if p.get("article") else ""),
            a=p["a"], b=p["b"], title_a=p["title_a"], title_b=p["title_b"],
            domain_a=p["domain_a"], domain_b=p["domain_b"],
            cov_a=cov(pa), cov_b=cov(pb),
            fields_a=fmt_fields(pa), fields_b=fmt_fields(pb),
            join_keys=", ".join(p["join_keys"]), passage_block=pb_block)

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if pairs:
            print("\n--- sample prompt ---\n" + prompt(pairs[0])[:2200])
        return
    if not pairs:
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(p)}]
             for p in pairs]
    t0 = time.time()
    try:
        outs = llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(convs, sampling)
    print(f"[INFO] generated in {(time.time()-t0)/60:.1f}m")

    out = Path(OUT_TMPL.format(tier=args.tier))
    ok = single = 0
    with out.open("w", encoding="utf-8") as fh:
        for p, o in zip(pairs, outs):
            obj = parse_json(o.outputs[0].text if o.outputs else "")
            if not obj or not obj.get("question"):
                continue
            if obj.get("single_table_answerable"):
                single += 1
                continue
            ok += 1
            fh.write(json.dumps({**obj, "tier": p["tier"], "evidence": p["evidence"],
                                 "answer_is_plot": obj.get("answer_shape", "").startswith("series"),
                                 "datasets": [{"code": p["a"], "title_en": p["title_a"],
                                               "domain": p["domain_a"]},
                                              {"code": p["b"], "title_en": p["title_b"],
                                               "domain": p["domain_b"]}],
                                 "join_keys_available": p["join_keys"],
                                 "source_article": p.get("article"),
                                 "source_url": p.get("article_url"),
                                 "passage": p.get("passage", "")[:400]},
                                ensure_ascii=False) + "\n")
    print(f"[DONE] {ok:,} two-hop questions ({single:,} rejected as single-table answerable) -> {out}")


if __name__ == "__main__":
    main()
