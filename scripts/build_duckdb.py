#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a persistent DuckDB over the cached CBS tables plus the full catalogue.

    python scripts/build_duckdb.py            # -> data/processed/cbs.duckdb
    duckdb data/processed/cbs.duckdb

Gives you three things in one database:
  catalog   all 4,868 CBS tables (id, title, status, observation count)
  enriched  the LLM enrichment: English title, description, topics
  t_<id>    one view per table whose observations are cached locally

Tables not cached are absent by design: the full CBS corpus is 13.4 billion
observations. Fetch one on demand with
    python -m cbs.fetch_table_data --table <id>
and re-run this script to pick it up.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import duckdb
import pandas as pd

TABLES = Path("data/processed/tables")
DB = Path("data/processed/cbs.duckdb")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a persistent DuckDB over CBS data")
    ap.add_argument("--db", default=DB, type=Path)
    ap.add_argument("--publisher", default="CBS")
    args = ap.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()
    con = duckdb.connect(str(args.db))

    cat = pd.read_parquet("data/processed/statline_catalog.parquet")
    cat = cat.rename(columns={"table_id": "table_id", "Title": "title",
                              "Status": "status", "ObservationCount": "n_observations"})
    con.execute("CREATE TABLE catalog AS SELECT * FROM cat")
    print(f"  catalog  : {len(cat):,} tables")

    rows = []
    with open("data/processed/enriched_unified_qwen3-32b.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("publisher") != args.publisher:
                continue
            rows.append({"code": r["code"], "title_en": r.get("title_en"),
                         "title_native": r.get("title_native"),
                         "description": r.get("enriched_description"),
                         "topics": ", ".join(r.get("topics") or []),
                         "coverage_start": (r.get("coverage") or {}).get("start"),
                         "coverage_end": (r.get("coverage") or {}).get("end"),
                         "n_observations": r.get("n_observations")})
    enr = pd.DataFrame(rows)
    con.execute("CREATE TABLE enriched AS SELECT * FROM enr")
    print(f"  enriched : {len(enr):,} datasets")

    n = 0
    for p in sorted(glob.glob(str(TABLES / "*.parquet"))):
        code = Path(p).stem
        view = "t_" + "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in code)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS "
                    f"SELECT * FROM read_parquet('{Path(p).resolve().as_posix()}')")
        n += 1
    print(f"  tables   : {n:,} views (observations cached locally)")
    con.close()
    print(f"[DONE] {args.db}  ({args.db.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
