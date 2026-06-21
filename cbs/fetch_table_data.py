#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and tidy the actual observation data for a CBS StatLine table.

Fetches every observation and joins human-readable labels for the measure and
for each dimension code (including period labels), producing an analysis-ready
"long" dataframe — so the doc2query example questions can actually be answered
and plotted.

    python -m cbs.fetch_table_data --table 82610NED
    -> data/processed/tables/82610NED.parquet  (+ .csv)

Reuses the OData v4 client + metadata fetch from cbs.odata_client.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from cbs.odata_client import CbsODataClient

OUT_DIR = Path("data/processed/tables")


def fetch_tidy(table_id: str, client: Optional[CbsODataClient] = None,
               max_obs: Optional[int] = None) -> pd.DataFrame:
    client = client or CbsODataClient(delay=0.0)
    meta = client.fetch_table_metadata(table_id)

    dims = [d.get("Identifier") for d in meta["dimensions"] if d.get("Identifier")]
    measure_title = {m.get("Identifier"): m.get("Title") for m in meta["measure_codes"]}
    measure_unit = {m.get("Identifier"): m.get("Unit") for m in meta["measure_codes"]}
    # code label lookups per dimension: {dim: {code: title}}
    code_label: Dict[str, Dict[str, str]] = {
        dim: {c.get("Identifier"): c.get("Title") for c in codes}
        for dim, codes in meta["codes"].items()
    }

    obs = client.fetch_observations(table_id, top=max_obs)
    rows = []
    for o in obs:
        row = {
            "measure_code": o.get("Measure"),
            "measure": measure_title.get(o.get("Measure"), o.get("Measure")),
            "unit": measure_unit.get(o.get("Measure")),
            "value": o.get("Value"),
            "value_text": (o.get("StringValue") or "").strip() or None,
            "status": o.get("ValueAttribute"),
        }
        for dim in dims:
            code = o.get(dim)
            row[dim] = code
            row[f"{dim}_label"] = code_label.get(dim, {}).get(code, code)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["title"] = meta["properties"].get("Title", table_id)
    df.attrs["dimensions"] = dims
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch + tidy a CBS table's observations")
    ap.add_argument("--table", required=True)
    ap.add_argument("--max-obs", type=int, default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR), type=Path)
    args = ap.parse_args()

    df = fetch_tidy(args.table, max_obs=args.max_obs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pq = args.out_dir / f"{args.table}.parquet"
    csv = args.out_dir / f"{args.table}.csv"
    df.to_parquet(pq, index=False)
    df.to_csv(csv, index=False)
    print(f"[OK] {args.table}: {len(df)} observations -> {pq}")
    print(f"     title: {df.attrs.get('title')}")
    print(f"     dimensions: {df.attrs.get('dimensions')}")
    print(f"     measures: {df['measure'].nunique()} | sample: {df['measure'].dropna().unique()[:5].tolist()}")


if __name__ == "__main__":
    main()
