#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Executable substrate for the benchmark: DuckDB views over statistical tables.

Every CBS table fetched by `cbs.fetch_table_data` lands in the same canonical
long schema, which is what makes one SQL dialect work across 12,308 datasets:

    measure_code, measure, unit, value, value_text, status,
    <Dim>, <Dim>_label            (one pair per dimension)

Eurostat tables are wide (one column per dimension + `value`) and are exposed as
they come; their schema is recorded per item.

Tables are materialised on demand and cached under data/processed/tables/.

    from bench.substrate import connect, view
    con = connect(); view(con, "70895ned")
    con.execute("SELECT ... FROM t_70895ned").df()
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

TABLES = Path("data/processed/tables")
CATALOG = Path("data/processed/statline_catalog.parquet")
# An uncached table is fetched from the API. CBS tables run to 1.9e9 observations,
# so an unbounded fetch can stall for hours - it once burned a 2-hour GPU job that
# produced no output at all. Cap what we pull, and refuse what is hopeless.
DEFAULT_MAX_OBS = 500_000
MAX_CATALOGUE_OBS = 5_000_000
DOWNLOADS = Path("downloads")
CBS_ID = re.compile(r"^\d{4,6}[A-Za-z]{2,4}$")


def connect():
    import duckdb
    return duckdb.connect()


def view_name(code: str) -> str:
    return "t_" + re.sub(r"[^0-9A-Za-z_]", "_", code)


def catalogue_size(code: str) -> Optional[int]:
    try:
        import pandas as pd
        cat = pd.read_parquet(CATALOG)
        row = cat[cat.table_id.astype(str) == str(code)]
        if len(row):
            v = pd.to_numeric(row.iloc[0].get("ObservationCount"), errors="coerce")
            return None if v != v else int(v)          # NaN check
    except Exception:  # noqa: BLE001
        pass
    return None


def _cbs_parquet(code: str, max_obs: Optional[int] = None) -> Path:
    path = TABLES / f"{code}.parquet"
    if not path.exists():
        n = catalogue_size(code)
        if n is not None and n > MAX_CATALOGUE_OBS:
            raise RuntimeError(
                f"{code} has {n:,} observations (> {MAX_CATALOGUE_OBS:,}); refusing to "
                f"materialise. Fetch a slice explicitly if you need it.")
        from cbs.fetch_table_data import fetch_tidy
        print(f"  [substrate] fetching {code}"
              + (f" ({n:,} observations, capped at {max_obs or DEFAULT_MAX_OBS:,})" if n else ""),
              flush=True)
        df = fetch_tidy(code, max_obs=max_obs or DEFAULT_MAX_OBS)
        TABLES.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return path


def _eurostat_csv(code: str) -> Optional[Path]:
    up = code.upper()
    hits = [p for p in DOWNLOADS.glob("*.csv") if p.stem.split("_")[0].upper() == up]
    return hits[0] if hits else None


def view(con, code: str, max_obs: Optional[int] = None) -> str:
    """Register `code` as a DuckDB view and return the view name."""
    name = view_name(code)
    if CBS_ID.match(code):
        src = _cbs_parquet(code, max_obs)
        con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                    f"SELECT * FROM read_parquet('{src.as_posix()}')")
        return name
    csv = _eurostat_csv(code)
    if csv is None:
        from eurostat_fetch_one import fetch_eurostat_dataset, flatten_sdmx_json
        df = flatten_sdmx_json(fetch_eurostat_dataset(code, {}, timeout=120, retries=2))
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        csv = DOWNLOADS / f"{code.upper()}_bench.csv"
        df.to_csv(csv, index=False)
    con.execute(f"CREATE OR REPLACE VIEW {name} AS "
                f"SELECT * FROM read_csv_auto('{csv.as_posix()}')")
    return name


def views(con, codes: Iterable[str]) -> list:
    return [view(con, c) for c in codes]
