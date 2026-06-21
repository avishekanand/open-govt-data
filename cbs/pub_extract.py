#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4 of the publication layer: LLM extraction of CBS data usage.

For each downloaded document, ask a local Ollama model which CBS (Statistics
Netherlands) datasets / microdata registers / StatLine tables the publication
actually used, and a one-line summary of how. Resumable.

Input : data/processed/pub/documents.parquet  (from cbs.pub_download)
Output: data/processed/pub/pub_extractions.parquet
        (merged into the Markdown by cbs.pub_report)

    python -m cbs.pub_extract --model qwen2.5:7b --limit 40
    python -m cbs.pub_extract --all                      # every OK document
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

PUB = Path("data/processed/pub")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("PUB_MODEL", "qwen2.5:7b")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

SYSTEM = (
    "You analyse a research publication and identify which Statistics Netherlands "
    "(CBS) data it used. CBS microdata registers include e.g. GBA/BRP (population), "
    "POLIS/SPOLIS (jobs & wages), SECMBUS (socio-economic status), HOOGSTEOPLTAB "
    "(education), INPATAB/INHATAB (income), VSLGWBTAB (neighbourhood), plus public "
    "StatLine aggregate tables. Return STRICT JSON only."
)
USER = """\
Publication title: {title}
CBS project number (if any): {project}

Text excerpt:
\"\"\"
{text}
\"\"\"

Return JSON:
{{
  "uses_cbs_data": true/false,
  "cbs_datasets": ["specific CBS datasets/registers/StatLine tables actually used; [] if none/unclear"],
  "data_kind": "microdata | aggregate/StatLine | both | unclear",
  "summary": "<one sentence: what CBS data was used and for what analysis>"
}}
"""


def ollama_json(text_prompt: str, model: str, timeout: int = 120) -> Dict[str, Any]:
    r = requests.post(
        OLLAMA_HOST.rstrip("/") + "/api/chat",
        json={"model": model,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": text_prompt}],
              "options": {"temperature": 0.1, "num_ctx": 8192},
              "stream": False, "format": "json"},
        timeout=timeout,
    )
    r.raise_for_status()
    c = r.json().get("message", {}).get("content", "")
    try:
        return json.loads(c)
    except Exception:
        m = JSON_RE.search(c or "")
        return json.loads(m.group(0)) if m else {}


def load_done(path: Path) -> set:
    if path.exists():
        try:
            return set(pd.read_parquet(path, columns=["url"])["url"])
        except Exception:
            return set()
    return set()


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-extract CBS datasets used per publication")
    ap.add_argument("--docs", default=str(PUB / "documents.parquet"), type=Path)
    ap.add_argument("--out", default=str(PUB / "pub_extractions.parquet"), type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="Process all OK docs (default: microdata/CBS-signal only)")
    ap.add_argument("--min-text", type=int, default=300)
    args = ap.parse_args()

    docs = pd.read_parquet(args.docs)
    docs = docs[docs["ok"] & (docs["text_len"] >= args.min_text)].copy()
    if not args.all:
        docs = docs[docs["mentions_microdata"] | (docs["n_cbs_table_ids"] > 0)]
    docs = docs.sort_values("text_len", ascending=False)

    done = load_done(args.out)
    docs = docs[~docs["url"].isin(done)]
    if args.limit:
        docs = docs.head(args.limit)
    print(f"[INFO] {len(docs)} documents to extract (model={args.model}, {len(done)} done)")

    rows: List[Dict[str, Any]] = []
    for i, (_, d) in enumerate(docs.iterrows(), 1):
        prompt = USER.format(title=(d.get("title") or "")[:200],
                             project=d.get("project"),
                             text=(d.get("text") or "")[:6000])
        try:
            o = ollama_json(prompt, args.model)
            ds = o.get("cbs_datasets") or []
            rows.append({
                "url": d["url"],
                "uses_cbs_data": bool(o.get("uses_cbs_data")),
                "cbs_datasets": ", ".join(ds) if isinstance(ds, list) else str(ds),
                "data_kind": o.get("data_kind", "unclear"),
                "summary": o.get("summary", ""),
            })
            print(f"[{i}/{len(docs)}] {d['domain']}: {', '.join(ds)[:60] or '(none)'}")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(docs)}] {d['url'][:50]} FAIL: {exc}")
        if i % 10 == 0 and rows:
            _flush(rows, args.out); rows = []
    _flush(rows, args.out)
    print(f"[DONE] extractions -> {args.out}")


def _flush(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


if __name__ == "__main__":
    main()
