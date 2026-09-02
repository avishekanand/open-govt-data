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

# Pulled observation data (cbs.fetch_table_data), used to ground the prompt in real values.
TABLES_DIR = Path("data/processed/tables")

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

ACTUAL category values present in the data (sampled from the observations):
{sample_values}

Period coverage: {period_coverage}

Observed measure value ranges:
{value_ranges}

IMPORTANT: base "example_queries" ONLY on categories, periods and measures that actually
appear above. Do not invent breakdowns (e.g. an age band or sector) that is not listed.
Where the real values are unavailable, keep the queries generic rather than guessing.

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


NO_OBS = {
    "sample_values": "(observation data not pulled for this table)",
    "period_coverage": "(unknown)",
    "value_ranges": "(observation data not pulled for this table)",
}


def build_observation_context(table_id: str, tables_dir: Path = TABLES_DIR,
                              max_values: int = 10, max_measures: int = 8) -> Dict[str, str]:
    """Real category labels / periods / value ranges from a pulled observation table.

    The ingested metadata only stores dimension *titles* (the code lists live behind
    a CodesUrl that is never fetched), so without this the model has to guess which
    breakdowns exist. Reads data/processed/tables/<table_id>.parquet when present.
    """
    path = Path(tables_dir) / f"{table_id}.parquet"
    if not path.exists():
        return dict(NO_OBS)
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - a corrupt/partial pull must not kill enrichment
        return dict(NO_OBS)

    lines: List[str] = []
    period = "(unknown)"
    for col in [c for c in df.columns if c.endswith("_label")]:
        dim = col[: -len("_label")]
        vals = [str(v).strip() for v in pd.unique(df[col].dropna()) if str(v).strip()]
        if not vals:
            continue
        shown = vals[:max_values]
        more = f" ... (+{len(vals) - len(shown)} more)" if len(vals) > len(shown) else ""
        lines.append(f"- {dim} ({len(vals)} categories): " + "; ".join(shown) + more)
        if dim.lower().startswith("perioden"):
            period = f"{vals[0]} ... {vals[-1]} ({len(vals)} periods)"

    ranges: List[str] = []
    if {"measure", "value"} <= set(df.columns):
        numeric = df.dropna(subset=["value"])
        for name, grp in list(numeric.groupby("measure", sort=False))[:max_measures]:
            unit = ""
            if "unit" in grp.columns:
                units = grp["unit"].dropna()
                if len(units):
                    unit = f" {units.iloc[0]}"
            ranges.append(f"- {name}: {grp['value'].min():.4g} to {grp['value'].max():.4g}{unit}")

    return {
        "sample_values": "\n".join(lines) if lines else NO_OBS["sample_values"],
        "period_coverage": period,
        "value_ranges": "\n".join(ranges) if ranges else "(no numeric values in this table)",
    }


def build_context(table_id: str, dims_df: pd.DataFrame, meas_df: pd.DataFrame,
                  max_measures: int = 25,
                  tables_dir: Optional[Path] = TABLES_DIR) -> Dict[str, str]:
    dims = dims_df[dims_df.table_id == table_id]["dimension_title"].dropna().tolist()
    meas = meas_df[meas_df.table_id == table_id]["title"].dropna().tolist()
    ctx = {
        "dimensions": ", ".join(dims) if dims else "(none)",
        "measures": ", ".join(meas[:max_measures]) if meas else "(none)",
    }
    ctx.update(build_observation_context(table_id, tables_dir) if tables_dir else dict(NO_OBS))
    return ctx


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
    ap.add_argument("--tables-dir", default=TABLES_DIR, type=Path,
                    help="Pulled observation parquets used to ground the prompt "
                         "(pass --no-observations to disable)")
    ap.add_argument("--no-observations", action="store_true",
                    help="Ignore pulled observation data; use metadata titles only")
    args = ap.parse_args()
    if args.no_observations:
        args.tables_dir = None

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
            ctx = build_context(tid, dims, meas, tables_dir=args.tables_dir)
            prompt = USER_TEMPLATE.format(
                table_id=tid,
                title=r.get("title") or "",
                description=(str(r.get("summary") or "")[:1500]),
                dimensions=ctx["dimensions"],
                measures=ctx["measures"],
                sample_values=ctx["sample_values"],
                period_coverage=ctx["period_coverage"],
                value_ranges=ctx["value_ranges"],
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
