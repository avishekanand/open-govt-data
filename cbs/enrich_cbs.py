#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
doc2query-style enrichment for CBS StatLine table metadata.

Mirrors the Eurostat enrichment (csv_to_ollama_jsonl_complete_only.py) but for
CBS tables. Reads the batch-ingested semantic layer:

    data/processed/catalog_meta/statline_datasets.parquet
    data/processed/catalog_meta/statline_dimensions.parquet
    data/processed/catalog_meta/statline_measures.parquet

For each table it builds a context (title + Dutch description + dimension titles
+ a sample of measure titles) and asks a local Ollama model (gemma4) to return
JSON with an English enriched description + example queries, so Dutch tables
become searchable in English. Output: one JSONL record per table.

Usage:
  python -m cbs.enrich_cbs --limit 8 --model gemma4:latest
  python -m cbs.enrich_cbs --output data/processed/cbs_enriched_gemma4.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a data librarian enriching metadata for Statistics Netherlands (CBS) "
    "StatLine tables so they are discoverable by English-speaking analysts. "
    "The source metadata is in Dutch. Return STRICT JSON only — no prose, no markdown."
)

USER_TEMPLATE = """\
Enrich this CBS StatLine table. The description is Dutch; write the enrichment in ENGLISH
but keep key Dutch domain terms in parentheses where helpful.

table_id: {table_id}
title (nl): {title}
description (nl): {description}
dimensions: {dimensions}
key measures (sample): {measures}

Return a JSON object with EXACTLY these keys:
{{
  "code": "{table_id}",
  "title_en": "<concise English title>",
  "enriched_description": "<2-4 sentence English description of what the table contains and how it is broken down>",
  "example_queries": ["<5-6 natural-language questions this table can answer, in English>"],
  "potential_applications": ["<3-4 concrete use cases>"],
  "key_dimensions": ["<the dimensions a user would filter by>"],
  "topics": ["<3-6 short English topic tags, e.g. population, income, health>"],
  "confidence": {{"desc": <0-1>, "queries": <0-1>}}
}}
"""

REQUIRED_KEYS = {
    "code", "title_en", "enriched_description", "example_queries",
    "potential_applications", "key_dimensions", "topics", "confidence",
}


def call_ollama(host: str, model: str, user_prompt: str,
                temperature: float, num_ctx: int, timeout: int, retries: int) -> str:
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": False,
        "format": "json",
    }
    backoff = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
            content = resp.json().get("message", {}).get("content", "")
            if not content:
                raise RuntimeError("empty content")
            return content
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(backoff)
            backoff *= 1.8
    return ""


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        m = JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


def build_context(table_id: str, dims_df: pd.DataFrame, meas_df: pd.DataFrame,
                  max_measures: int = 25) -> Dict[str, str]:
    dims = dims_df[dims_df.table_id == table_id]["dimension_title"].dropna().tolist()
    meas = meas_df[meas_df.table_id == table_id]["title"].dropna().tolist()
    return {
        "dimensions": ", ".join(dims) if dims else "(none)",
        "measures": ", ".join(meas[:max_measures]) if meas else "(none)",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="doc2query enrichment for CBS StatLine metadata")
    ap.add_argument("--meta-dir", default="data/processed/catalog_meta", type=Path)
    ap.add_argument("--output", default="data/processed/cbs_enriched_gemma4.jsonl", type=Path)
    ap.add_argument("--model", default="gemma4:latest")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                    help="Ollama host (default: $OLLAMA_HOST or http://localhost:11434)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_ctx", type=int, default=8192)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="Enrich at most N tables")
    ap.add_argument("--resume", action="store_true", help="Skip tables already in output")
    args = ap.parse_args()

    datasets = pd.read_parquet(args.meta_dir / "statline_datasets.parquet")
    dims = pd.read_parquet(args.meta_dir / "statline_dimensions.parquet")
    meas = pd.read_parquet(args.meta_dir / "statline_measures.parquet")

    done: set = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["code"])
            except Exception:
                pass

    rows = datasets.drop_duplicates("table_id")
    if args.limit:
        rows = rows.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.output.exists()) else "w"
    ok = fail = skip = 0
    t0 = time.time()
    with args.output.open(mode, encoding="utf-8") as fout:
        for i, (_, r) in enumerate(rows.iterrows(), 1):
            tid = r["table_id"]
            if tid in done:
                skip += 1
                continue
            ctx = build_context(tid, dims, meas)
            prompt = USER_TEMPLATE.format(
                table_id=tid,
                title=r.get("title") or "",
                description=(str(r.get("summary") or "")[:1500]),
                dimensions=ctx["dimensions"],
                measures=ctx["measures"],
            )
            try:
                raw = call_ollama(args.host, args.model, prompt, args.temperature,
                                  args.num_ctx, args.timeout, args.retries)
                obj = parse_json(raw)
                if not obj or REQUIRED_KEYS - set(obj.keys()):
                    raise ValueError(f"missing keys: {REQUIRED_KEYS - set(obj.keys()) if obj else 'unparseable'}")
                obj["code"] = tid
                obj["title_nl"] = r.get("title")
                obj["source_url"] = r.get("source_url")
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                fout.flush()
                ok += 1
                dt = time.time() - t0
                print(f"[{i}/{len(rows)}] {tid} OK  ({dt/i:.1f}s/table)  {obj.get('title_en','')[:50]}")
            except Exception as exc:  # noqa: BLE001
                fail += 1
                print(f"[{i}/{len(rows)}] {tid} FAIL: {exc}")

    print(f"\n[DONE] enriched ok={ok} fail={fail} skip={skip} -> {args.output} "
          f"in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
