#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the classified questions into per-publisher seed sets for the benchmark.

    python scripts/build_seed_sets.py

Writes, under data/processed/benchmark/:
    seeds_cbs.jsonl     verifiable_now questions whose data would come from CBS
    seeds_estat.jsonl   ... from Eurostat
    seeds_rejected.jsonl everything filtered out, WITH the reason (auditable)
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path("data/processed/pub/pub_evidence.jsonl")
OUT = Path("data/processed/benchmark")
SINGLE_VALUED = {"single_number", "rate_or_share", "comparison", "ranking_or_list", "yes_no"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in SRC.open(encoding="utf-8")]
    buckets = {"CBS": [], "ESTAT": [], "rejected": []}

    for r in recs:
        linked = [x["link"] for x in r["data_mentions"]
                  if x["link"]["match"] in ("exact_id", "fuzzy_title")]
        registers = sorted({x["link"]["code"] for x in r["data_mentions"]
                            if x["link"]["match"] == "register"})
        for q in r.get("research_questions") or []:
            item = {
                "question_en": q.get("question_en"),
                "question_native": q.get("question"),
                "witness": q.get("witness"),
                "witness_verified": q.get("witness_verified"),
                "answer_type": q.get("answer_type"),
                "answer_shape": ("single_valued" if q.get("answer_type") in SINGLE_VALUED
                                 else "series_or_plot"),
                "data_needed": q.get("data_needed"),
                "publisher_hint": q.get("publisher_hint"),
                "verifiable_now": q.get("verifiable_now"),
                "classifier_reason": q.get("reason"),
                "source_url": r.get("final_url") or r.get("url"),
                "source_title": r.get("title"),
                "source_domain": r.get("domain"),
                "candidate_datasets": [{"code": x["code"], "title": x["title"],
                                        "publisher": x["publisher"], "match": x["match"]}
                                       for x in linked],
                "microdata_registers": registers,
            }
            if not q.get("verifiable_now"):
                item["rejected_because"] = (
                    f"not verifiable from public aggregates ({q.get('data_needed')}, "
                    f"{q.get('answer_type')})")
                buckets["rejected"].append(item)
            elif q.get("publisher_hint") == "ESTAT":
                buckets["ESTAT"].append(item)
            elif q.get("publisher_hint") in ("CBS", "either"):
                buckets["CBS"].append(item)
            else:
                item["rejected_because"] = "verifiable but no publisher could be assigned"
                buckets["rejected"].append(item)

    for name, path in (("CBS", OUT / "seeds_cbs.jsonl"), ("ESTAT", OUT / "seeds_estat.jsonl"),
                       ("rejected", OUT / "seeds_rejected.jsonl")):
        with path.open("w", encoding="utf-8") as fh:
            for it in buckets[name]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        sv = sum(1 for it in buckets[name] if it["answer_shape"] == "single_valued")
        wd = sum(1 for it in buckets[name] if it["candidate_datasets"])
        print(f"{path}  {len(buckets[name]):5,}  single-valued {sv:4,}  with linked dataset {wd:3,}")


if __name__ == "__main__":
    main()
