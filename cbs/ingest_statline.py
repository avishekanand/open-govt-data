#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest a single CBS StatLine table into normalized Parquet tables.

Foundation slice of the "Dutch Public Data Intelligence Engine":
  - fetch the full semantic layer (Properties / Dimensions / Codes / Measures)
  - fetch a SMALL observations sample (default: Amsterdam + Delft)
  - normalize + join human-readable labels
  - persist raw JSON (provenance) and processed Parquet

Outputs (under --processed-dir, default data/processed/):
  statline_datasets.parquet
  statline_dimensions.parquet
  statline_codes.parquet
  statline_measures.parquet
  statline_observations_sample.parquet

Raw JSON is written under --raw-dir (default data/raw/cbs/<TABLE_ID>/).

Usage:
  python -m cbs.ingest_statline --table 83765NED
  python -m cbs.ingest_statline --table 83765NED --regions GM0363 GM0503
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cbs.odata_client import CBS_BASE, CbsODataClient

# Default sample regions: Amsterdam, Delft (municipality codes, CBS v4).
DEFAULT_REGIONS = ["GM0363", "GM0503"]
# The geo dimension differs per table; we resolve it dynamically but fall back
# to this common one for neighbourhood/municipality tables.
GEO_DIM_HINTS = ("WijkenEnBuurten", "RegioS", "Regios")


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stringify nested dicts/lists so Parquet (arrow) stays happy.
    safe = df.copy()
    for col in safe.columns:
        if safe[col].apply(lambda x: isinstance(x, (dict, list))).any():
            safe[col] = safe[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x
            )
    safe.to_parquet(path, index=False)


def _dump_raw(meta: Dict[str, Any], observations: List[Dict[str, Any]], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (raw_dir / "observations_sample.json").write_text(
        json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def resolve_geo_dim(dimensions: List[Dict[str, Any]]) -> Optional[str]:
    """Pick the geographic/region dimension for sampling, if any."""
    ids = [d.get("Identifier") for d in dimensions]
    for hint in GEO_DIM_HINTS:
        if hint in ids:
            return hint
    # Otherwise: first dimension flagged as a geo detail kind.
    for d in dimensions:
        if "Geo" in str(d.get("Kind", "")):
            return d.get("Identifier")
    return None


# --------------------------------------------------------------- normalizers
def normalize_dataset(table_id: str, props: Dict[str, Any]) -> pd.DataFrame:
    row = {
        "table_id": table_id,
        "title": props.get("Title"),
        "short_title": props.get("ShortTitle"),
        "summary": props.get("Summary") or props.get("Description"),
        "language": props.get("Language"),
        "catalog": props.get("Catalog"),
        "status": props.get("Status"),
        "modified_at": props.get("Modified"),
        "frequency": props.get("Frequency"),
        "period": props.get("Period"),
        "source_url": f"{CBS_BASE}/{table_id}",
        "raw_metadata": props,
    }
    return pd.DataFrame([row])


def normalize_dimensions(table_id: str, dimensions: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "table_id": table_id,
            "dimension_key": d.get("Identifier"),
            "dimension_title": d.get("Title"),
            "dimension_type": d.get("Kind"),
            "contains_codes": d.get("ContainsCodes"),
            "contains_groups": d.get("ContainsGroups"),
            "raw": d,
        }
        for d in dimensions
    ]
    return pd.DataFrame(rows)


def normalize_codes(table_id: str, codes_by_dim: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows = []
    for dim, codes in codes_by_dim.items():
        for c in codes:
            rows.append(
                {
                    "table_id": table_id,
                    "dimension_key": dim,
                    "code": c.get("Identifier"),
                    "title": c.get("Title"),
                    "description": c.get("Description"),
                    "parent_code": c.get("ParentId") or c.get("DimensionGroupId"),
                    "raw": c,
                }
            )
    return pd.DataFrame(rows)


def normalize_measures(
    table_id: str,
    measure_codes: List[Dict[str, Any]],
    measure_groups: List[Dict[str, Any]],
) -> pd.DataFrame:
    group_title = {g.get("Id"): g.get("Title") for g in measure_groups}
    rows = [
        {
            "table_id": table_id,
            "measure_code": m.get("Identifier"),
            "title": m.get("Title"),
            "group_id": m.get("MeasureGroupId"),
            "group_title": group_title.get(m.get("MeasureGroupId")),
            "unit": m.get("Unit"),
            "data_type": m.get("DataType"),
            "decimals": m.get("Decimals"),
            "description": m.get("Description"),
            "raw": m,
        }
        for m in measure_codes
    ]
    return pd.DataFrame(rows)


def normalize_observations(
    table_id: str,
    observations: List[Dict[str, Any]],
    geo_dim: Optional[str],
    measure_title: Dict[str, str],
    region_title: Dict[str, str],
) -> pd.DataFrame:
    rows = []
    for o in observations:
        region_code = o.get(geo_dim) if geo_dim else None
        rows.append(
            {
                "table_id": table_id,
                "measure_code": o.get("Measure"),
                "measure_title": measure_title.get(o.get("Measure")),
                "region_code": region_code,
                "region_title": region_title.get(region_code) if region_code else None,
                "value_numeric": o.get("Value"),
                "value_text": (o.get("StringValue") or "").strip() or None,
                "status_flag": o.get("ValueAttribute"),
                "geo_dimension": geo_dim,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- driver
def ingest_table(
    table_id: str,
    regions: List[str],
    processed_dir: Path,
    raw_dir: Path,
    sample_top: Optional[int] = None,
    client: Optional[CbsODataClient] = None,
) -> Dict[str, pd.DataFrame]:
    client = client or CbsODataClient(delay=0.1)

    print(f"[INFO] Fetching metadata for {table_id} ...")
    meta = client.fetch_table_metadata(table_id)
    print(f"[INFO]   entities: {meta['entities']}")
    print(f"[INFO]   dimensions: {[d.get('Identifier') for d in meta['dimensions']]}")
    print(f"[INFO]   measures: {len(meta['measure_codes'])}")

    geo_dim = resolve_geo_dim(meta["dimensions"])
    print(f"[INFO]   geo dimension: {geo_dim}")

    # Sample observations.
    if geo_dim and regions:
        print(f"[INFO] Fetching observations for {geo_dim} in {regions} ...")
        observations = client.fetch_observations_for(table_id, geo_dim, regions, top=sample_top)
    else:
        print(f"[INFO] No geo dimension/regions; fetching top {sample_top or 1000} observations ...")
        observations = client.fetch_observations(table_id, top=sample_top or 1000)
    print(f"[INFO]   observations fetched: {len(observations)}")

    # Build label lookups for joins.
    measure_title = {m.get("Identifier"): m.get("Title") for m in meta["measure_codes"]}
    region_title: Dict[str, str] = {}
    if geo_dim and geo_dim in meta["codes"]:
        region_title = {c.get("Identifier"): c.get("Title") for c in meta["codes"][geo_dim]}

    frames = {
        "statline_datasets": normalize_dataset(table_id, meta["properties"]),
        "statline_dimensions": normalize_dimensions(table_id, meta["dimensions"]),
        "statline_codes": normalize_codes(table_id, meta["codes"]),
        "statline_measures": normalize_measures(
            table_id, meta["measure_codes"], meta["measure_groups"]
        ),
        "statline_observations_sample": normalize_observations(
            table_id, observations, geo_dim, measure_title, region_title
        ),
    }

    # Persist raw provenance + processed parquet.
    _dump_raw(meta, observations, raw_dir / table_id)
    for name, df in frames.items():
        out = processed_dir / f"{name}.parquet"
        _write_parquet(df, out)
        print(f"[OK] {name}: {len(df)} rows -> {out}")

    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a CBS StatLine table to Parquet")
    ap.add_argument("--table", default="83765NED", help="CBS table id (default 83765NED)")
    ap.add_argument(
        "--regions",
        nargs="*",
        default=DEFAULT_REGIONS,
        help="Region codes to sample observations for (default Amsterdam+Delft)",
    )
    ap.add_argument("--processed-dir", default="data/processed", type=Path)
    ap.add_argument("--raw-dir", default="data/raw/cbs", type=Path)
    ap.add_argument("--sample-top", type=int, default=None, help="Cap sample observations")
    args = ap.parse_args()

    ingest_table(
        table_id=args.table,
        regions=args.regions,
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        sample_top=args.sample_top,
    )
    print("\n[DONE] StatLine ingestion complete.")


if __name__ == "__main__":
    main()
