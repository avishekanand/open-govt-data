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
DOWNLOADS = Path("downloads")
CBS_ID = re.compile(r"^\d{4,6}[A-Za-z]{2,4}$")


def connect():
    import duckdb
    return duckdb.connect()


def view_name(code: str) -> str:
    return "t_" + re.sub(r"[^0-9A-Za-z_]", "_", code)


def _cbs_parquet(code: str, max_obs: Optional[int] = None) -> Path:
    path = TABLES / f"{code}.parquet"
    if not path.exists():
        from cbs.fetch_table_data import fetch_tidy
        df = fetch_tidy(code, max_obs=max_obs)
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
