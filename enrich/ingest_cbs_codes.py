#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harvest CBS dimension code lists - metadata only, no observations.

The CBS analogue of enrich.ingest_eurostat_meta. Without this, CBS category
values could only come from downloaded observation tables, so only the 795
tables whose data happened to be pulled were grounded, versus 7,438 on the
Eurostat side. The OData API serves each dimension's code list directly.

    python -m enrich.ingest_cbs_codes --workers 8
    -> data/processed/cbs_codelists.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd

from cbs.odata_client import CbsODataClient

DIMS = Path("data/processed/catalog_meta/statline_dimensions.parquet")
OUT = Path("data/processed/cbs_codelists.jsonl")
MAX_CODES = 400  # keep the biggest codelists bounded (some run to 15k regions)

_lock = threading.Lock()


def codes_for_table(client: CbsODataClient, table_id: str, dims: list) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dim_key, dim_title, has_codes in dims:
        if not has_codes:
            continue
        try:
            codes = client.fetch_dimension_codes(table_id, dim_key)
        except Exception:  # noqa: BLE001 - one bad dimension must not lose the table
            continue
        titles = [c.get("Title") for c in codes if c.get("Title")]
        out[dim_key] = {"name": dim_title or dim_key,
                        "n_categories": len(titles),
                        "categories": titles[:MAX_CODES]}
    return {"table_id": table_id, "dimensions": out}


def done_ids(path: Path) -> Set[str]:
    done = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["table_id"])
                except Exception:  # noqa: BLE001
                    pass
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest CBS dimension code lists")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT, type=Path)
    args = ap.parse_args()

    dm = pd.read_parquet(DIMS)
    by_table: Dict[str, list] = {}
    for r in dm.itertuples():
        by_table.setdefault(str(r.table_id), []).append(
            (r.dimension_key, r.dimension_title, bool(r.contains_codes)))

    done = done_ids(args.out)
    tables = [t for t in by_table if t not in done]
    if args.limit:
        tables = tables[: args.limit]
    print(f"[INFO] {len(tables):,} tables to harvest ({len(done):,} done), "
          f"{sum(len(by_table[t]) for t in tables):,} dimensions", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    client = CbsODataClient(delay=0.0)
    ok = fail = 0
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as fout, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(codes_for_table, client, t, by_table[t]): t for t in tables}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
                with _lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
            if i % 250 == 0:
                fout.flush()
                el = time.time() - t0
                print(f"[{i}/{len(tables)}] ok={ok} fail={fail} | {el/60:.1f}m "
                      f"(eta {(len(tables)-i)/max(i/el,1e-9)/60:.0f}m)", flush=True)
    print(f"[DONE] ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}m -> {args.out}")


if __name__ == "__main__":
    main()
