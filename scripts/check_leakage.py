#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure whether derived questions leak their gold dataset's vocabulary.

A question written from a table's schema tends to reuse that table's wording,
which would let lexical matching alone solve the benchmark. This measures the
overlap that the prompt merely asks for, rather than trusting it - and compares
against the ATTESTED questions, which cannot leak because they were written by
researchers who had never seen our catalogue.

    python scripts/check_leakage.py
"""
from __future__ import annotations

import json
import re
import statistics as st
from pathlib import Path

DERIVED = Path("data/processed/benchmark/derived_questions_cbs.jsonl")
VERIFIED = Path("data/processed/benchmark/question_dataset_verified_cbs_top10.jsonl")
WORD = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "on", "by", "in", "for", "and", "to", "from", "with",
        "per", "what", "which", "how", "many", "is", "are", "was", "were", "did", "do",
        "cbs", "statline", "netherlands", "dutch"}


def toks(s):
    return {w for w in WORD.findall((s or "").lower()) if w not in STOP}


def overlap(q, title):
    a, b = toks(q), toks(title)
    return len(a & b) / len(b) if b else 0.0


def main() -> None:
    if not DERIVED.exists():
        print(f"{DERIVED} not present yet — run enrich.derive_questions first")
        return
    rows = [json.loads(l) for l in DERIVED.open(encoding="utf-8")]
    dv = [overlap(r["derived_question"], (r["dataset"].get("title_en") or "")) for r in rows]
    ov = [overlap(r["original_question"], (r["dataset"].get("title_en") or "")) for r in rows]
    print(f"derived questions: {len(rows):,}")
    print(f"  overlap with gold dataset title — derived : median {st.median(dv):.2f}  "
          f"mean {st.mean(dv):.2f}  >0.5: {sum(1 for x in dv if x > .5)}")
    print(f"  overlap with gold dataset title — original: median {st.median(ov):.2f}  "
          f"mean {st.mean(ov):.2f}  >0.5: {sum(1 for x in ov if x > .5)}")
    delta = st.median(dv) - st.median(ov)
    print(f"\n  leakage introduced by derivation: {delta:+.2f} median overlap")
    print("  (attested questions are the floor: researchers never saw our catalogue)")
    if delta > 0.15:
        print("  VERDICT: derived questions leak noticeably; paraphrase or reject before use.")
    else:
        print("  VERDICT: derivation adds little vocabulary leakage.")
    print("\n  serves_original:",
          {k: sum(1 for r in rows if r.get("serves_original") == k)
           for k in ("yes", "partly", "no")})
    print("\n--- examples with the highest overlap (worst cases) ---")
    for r, o in sorted(zip(rows, dv), key=lambda t: -t[1])[:3]:
        print(f"  overlap {o:.2f} | {r['derived_question'][:88]}")
        print(f"             gold: {r['dataset'].get('title_en')}")


if __name__ == "__main__":
    main()
