#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata-only ingest of the FULL live Eurostat catalogue.

No observation download. For each dataset we fetch a 1-period slice, which
carries the complete structure - dataset label, dimension names, and every
category code->label - for a few KB instead of the full table. (Verified
against fully-pulled data for 83 datasets: zero categories missing.)

    python -m enrich.ingest_eurostat_meta --workers 8

Outputs
    data/processed/eurostat_catalog.parquet   live TOC (title, code, coverage, size)
    data/processed/eurostat_metadata.jsonl    one structure record per dataset
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eurostat_fetch_one import fetch_eurostat_dataset  # noqa: E402

TOC_URL = "https://ec.europa.eu/eurostat/api/dissemination/catalogue/toc/txt"
CATALOG = Path("data/processed/eurostat_catalog.parquet")
META = Path("data/processed/eurostat_metadata.jsonl")

_lock = threading.Lock()


def fetch_toc(timeout: int = 180) -> pd.DataFrame:
    r = requests.get(TOC_URL, timeout=timeout, headers={"User-Agent": "ogd-metadata/1.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t", quotechar='"', dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip().replace({"": None, "nan": None})
    df = df[df["type"].isin(["dataset", "table"])].drop_duplicates("code").reset_index(drop=True)
    df = df.rename(columns={"last update of data": "last_update",
                            "last table structure change": "last_structure_change",
                            "data start": "data_start", "data end": "data_end",
                            "values": "n_observations"})
    df["n_observations"] = pd.to_numeric(df["n_observations"], errors="coerce")
    return df


def structure_for(code: str) -> Dict[str, Any]:
    """Dataset label + dimension names + full category labels, from a 1-period slice."""
    data = fetch_eurostat_dataset(code, {"lastTimePeriod": "1"}, timeout=90, retries=2)
    dim_obj = data.get("dimension", {}) or {}
    ids = dim_obj.get("id") or [k for k, v in dim_obj.items()
                                if isinstance(v, dict) and "category" in v]
    dims = {}
    for dim in ids:
        d = dim_obj.get(dim, {}) or {}
        cat = d.get("category", {}) or {}
        dims[dim] = {"name": d.get("label") or dim,
                     "categories": cat.get("label", {}) or {}}
    return {"code": code, "dataset_label": data.get("label", ""), "dimensions": dims}


def done_codes(path: Path) -> Set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["code"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Metadata-only ingest of the Eurostat catalogue")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--catalog", default=CATALOG, type=Path)
    ap.add_argument("--out", default=META, type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    toc = fetch_toc()
    toc.to_parquet(args.catalog, index=False)
    print(f"[INFO] live catalogue: {len(toc):,} datasets -> {args.catalog}", flush=True)

    done = done_codes(args.out)
    codes = [c for c in toc["code"].dropna().tolist() if c not in done]
    if args.limit:
        codes = codes[: args.limit]
    print(f"[INFO] {len(codes):,} to fetch ({len(done):,} already done), {args.workers} workers",
          flush=True)

    ok = fail = 0
    failures = []
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(structure_for, c): c for c in codes}
        for i, fut in enumerate(as_completed(futs), 1):
            code = futs[fut]
            try:
                rec = fut.result()
                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                failures.append({"code": code, "error": f"{type(exc).__name__}: {exc}"[:200]})
            if i % 250 == 0:
                fout.flush()
                el = time.time() - t0
                print(f"[{i}/{len(codes)}] ok={ok} fail={fail} | {el/60:.1f}m "
                      f"({i/max(el,1):.1f}/s, eta {(len(codes)-i)/max(i/el,1e-9)/60:.0f}m)",
                      flush=True)
    dt = time.time() - t0
    print(f"[DONE] ok={ok} fail={fail} in {dt/60:.1f}m -> {args.out} "
          f"({args.out.stat().st_size/1e6:.1f} MB)")
    if failures:
        fp = args.out.with_suffix(".failures.json")
        fp.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] {len(failures)} failed -> {fp}")


if __name__ == "__main__":
    main()
