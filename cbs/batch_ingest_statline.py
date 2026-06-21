#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-ingest StatLine semantic metadata for many CBS tables.

Reads the catalogue (statline_catalog.parquet), selects a subset (default: the
N most-recently-updated active 'Regulier' tables), and for each table fetches
the core semantic layer:

    Properties      -> statline_datasets.parquet   (one row per table)
    Dimensions      -> statline_dimensions.parquet
    MeasureCodes(+Groups) -> statline_measures.parquet

Per-dimension CODE lists are intentionally skipped in bulk (they explode for
geo tables — fetch them on demand with cbs.ingest_statline for a single table).

Resumable: tables already present in the output are skipped. Writes
incrementally every --flush-every tables so a long run is crash-safe.

Usage:
  python -m cbs.batch_ingest_statline --limit 700
  python -m cbs.batch_ingest_statline --limit 700 --sample-data 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from cbs.odata_client import CBS_BASE, CbsODataClient
from cbs.ingest_statline import (
    normalize_dataset,
    normalize_dimensions,
    normalize_measures,
    normalize_observations,
    resolve_geo_dim,
)


def _load_done(path: Path) -> Set[str]:
    if path.exists():
        try:
            return set(pd.read_parquet(path)["table_id"].astype(str))
        except Exception:
            return set()
    return set()


def _append_parquet(new: pd.DataFrame, path: Path) -> None:
    if new.empty:
        return
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)


def select_tables(catalog_path: Path, limit: int, status: str) -> List[str]:
    cat = pd.read_parquet(catalog_path)
    sub = cat[cat["Status"] == status].copy()
    sort_col = "ObservationsModified" if "ObservationsModified" in sub.columns else "Modified"
    sub = sub.sort_values(sort_col, ascending=False)
    return sub["table_id"].astype(str).head(limit).tolist()


def ingest_metadata_only(client: CbsODataClient, table_id: str) -> Dict[str, pd.DataFrame]:
    """Fetch Properties + Dimensions + Measures for one table (no code lists).

    Entity sets vary per table (e.g. some lack MeasureGroups), so we only
    request what the table actually advertises.
    """
    entities = set(client.list_entities(table_id))
    props = client.fetch_properties(table_id) if "Properties" in entities else {}
    dims = client.fetch_dimensions(table_id) if "Dimensions" in entities else []
    mcodes = client.fetch_measure_codes(table_id) if "MeasureCodes" in entities else []
    mgroups = client.fetch_measure_groups(table_id) if "MeasureGroups" in entities else []
    return {
        "datasets": normalize_dataset(table_id, props),
        "dimensions": normalize_dimensions(table_id, dims),
        "measures": normalize_measures(table_id, mcodes, mgroups),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-ingest CBS StatLine metadata")
    ap.add_argument("--catalog", default="data/processed/statline_catalog.parquet", type=Path)
    ap.add_argument("--out-dir", default="data/processed/catalog_meta", type=Path)
    ap.add_argument("--status", default="Regulier", help="Catalogue status to select")
    ap.add_argument("--limit", type=int, default=700, help="Number of tables")
    ap.add_argument("--delay", type=float, default=0.1, help="Polite delay between tables")
    ap.add_argument("--flush-every", type=int, default=25, help="Persist every N tables")
    ap.add_argument("--sample-data", type=int, default=0,
                    help="After metadata, pull a tiny observation sample for the first K tables")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "datasets": out / "statline_datasets.parquet",
        "dimensions": out / "statline_dimensions.parquet",
        "measures": out / "statline_measures.parquet",
    }

    table_ids = select_tables(args.catalog, args.limit, args.status)
    done = _load_done(paths["datasets"])
    todo = [t for t in table_ids if t not in done]
    print(f"[INFO] selected {len(table_ids)} tables; {len(done)} already done; {len(todo)} to fetch")

    client = CbsODataClient(delay=args.delay)
    buffers: Dict[str, List[pd.DataFrame]] = {k: [] for k in paths}
    ok = fail = 0
    failures: List[str] = []
    t0 = time.time()

    def flush() -> None:
        for key, path in paths.items():
            if buffers[key]:
                _append_parquet(pd.concat(buffers[key], ignore_index=True), path)
                buffers[key] = []

    for i, tid in enumerate(todo, 1):
        try:
            frames = ingest_metadata_only(client, tid)
            buffers["datasets"].append(frames["datasets"])
            buffers["dimensions"].append(frames["dimensions"])
            buffers["measures"].append(frames["measures"])
            ok += 1
            if i % 10 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                eta = (len(todo) - i) / rate if rate else 0
                print(f"[{i}/{len(todo)}] {tid} ok  ({rate:.1f}/s, ETA {eta/60:.1f}m)  ok={ok} fail={fail}")
        except Exception as exc:  # noqa: BLE001
            fail += 1
            failures.append(f"{tid}: {exc}")
            print(f"[{i}/{len(todo)}] {tid} FAILED: {exc}")
        if i % args.flush_every == 0:
            flush()
    flush()

    print(f"\n[DONE] metadata: ok={ok} fail={fail} of {len(todo)} in {(time.time()-t0)/60:.1f}m")
    for f in failures[:20]:
        print("  fail:", f)

    # Optional: pull a small observation sample for the first K tables, to check.
    if args.sample_data > 0:
        print(f"\n[INFO] pulling observation samples for first {args.sample_data} tables ...")
        sample_path = out / "statline_observations_check.parquet"
        sample_frames = []
        for tid in table_ids[: args.sample_data]:
            try:
                dims = client.fetch_dimensions(tid)
                geo = resolve_geo_dim(dims)
                obs = client.fetch_observations(tid, top=20)
                mtitle = {m.get("Identifier"): m.get("Title") for m in client.fetch_measure_codes(tid)}
                df = normalize_observations(tid, obs, geo, mtitle, {})
                sample_frames.append(df)
                print(f"  {tid}: {len(df)} obs (geo={geo})")
            except Exception as exc:  # noqa: BLE001
                print(f"  {tid}: sample FAILED: {exc}")
        if sample_frames:
            pd.concat(sample_frames, ignore_index=True).to_parquet(sample_path, index=False)
            print(f"[OK] observation check -> {sample_path}")


if __name__ == "__main__":
    main()
