#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch the CBS StatLine table catalogue (v4 datasets API).

The catalogue is the backbone for dataset discovery: which official CBS
aggregate tables exist, with titles, descriptions, dates, status and
observation counts. One paginated call set — cheap.

    https://datasets.cbs.nl/odata/v1/CBS/Datasets

Output: data/processed/statline_catalog.parquet

Usage:
  python -m cbs.catalog
  python -m cbs.catalog --status Regulier          # active tables only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from cbs.odata_client import CBS_BASE, CbsODataClient

CATALOG_FIELDS = [
    "Identifier", "Title", "Description", "Language", "Catalog",
    "Status", "DatasetType", "Modified", "ReleaseDate", "ModificationDate",
    "ObservationsModified", "ObservationCount", "Version",
]


def fetch_catalog(
    client: Optional[CbsODataClient] = None,
    status: Optional[str] = None,
) -> pd.DataFrame:
    client = client or CbsODataClient(delay=0.0)
    url = f"{CBS_BASE}/Datasets"
    if status:
        url += f"?$filter=Status eq '{status}'"
    rows = client.get_odata(url)
    df = pd.DataFrame(rows)
    keep = [c for c in CATALOG_FIELDS if c in df.columns]
    df = df[keep].rename(columns={"Identifier": "table_id"})
    df["source_url"] = df["table_id"].map(lambda t: f"{CBS_BASE}/{t}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch CBS StatLine catalogue")
    ap.add_argument("--status", default=None, help="Filter by Status (e.g. Regulier)")
    ap.add_argument("--out", default="data/processed/statline_catalog.parquet", type=Path)
    args = ap.parse_args()

    df = fetch_catalog(status=args.status)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[OK] catalogue: {len(df)} tables -> {args.out}")
    if "Status" in df.columns:
        print("[INFO] by status:", df["Status"].value_counts().to_dict())
    if "DatasetType" in df.columns:
        print("[INFO] by type:", df["DatasetType"].value_counts().to_dict())


if __name__ == "__main__":
    main()
