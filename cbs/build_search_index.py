#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a SQLite FTS5 search index over CBS StatLine table metadata.

Flattens every text metadata field per table into one searchable document and
indexes it with FTS5 (BM25 term matching). Sources:

    data/processed/catalog_meta/statline_datasets.parquet    (one row/table)
    data/processed/catalog_meta/statline_dimensions.parquet
    data/processed/catalog_meta/statline_measures.parquet
    data/processed/cbs_enriched_gemma4.jsonl                 (gemma4 doc2query)
    data/processed/statline_catalog.parquet                  (ObservationCount)

Output: data/processed/cbs_search.db  with:
    tables       - structured row per table (display + raw text columns)
    tables_fts   - FTS5 virtual table (external content) over the text columns

Usage:
  python -m cbs.build_search_index
  python -m cbs.build_search_index --selftest
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List

import pandas as pd

from cbs.odata_client import CBS_BASE

PROC = Path("data/processed")
META = PROC / "catalog_meta"
DB_PATH = PROC / "cbs_search.db"

# Columns exposed to FTS5 (the searchable text).
FTS_COLS = [
    "title_nl", "title_en", "summary", "enriched_description",
    "example_queries", "topics", "dimensions_text", "measures_text",
]
MEASURE_DESC_CAP = 60  # cap measure descriptions concatenated per table (size guard)


def _join_unique(series: pd.Series) -> str:
    seen, out = set(), []
    for v in series.dropna().astype(str):
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return ", ".join(out)


def load_enrichment(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            out[str(r.get("code"))] = r
        except Exception:
            continue
    return out


def _conf(r: dict) -> float:
    """Self-reported confidence used to break ties when merging enrichment files."""
    c = r.get("confidence") or {}
    try:
        return float(c.get("desc", 0)) + float(c.get("queries", 0))
    except Exception:
        return 0.0


def resolve_enrichment_files(paths=None) -> List[Path]:
    """Default: every data/processed/cbs_enriched_*.jsonl, so each model/machine can
    write its own file and they all get merged. Or an explicit list of files."""
    if paths:
        return [Path(p) for p in paths]
    return sorted(PROC.glob("cbs_enriched_*.jsonl"))


def load_enrichments(paths: List[Path]) -> Dict[str, dict]:
    """Merge several enrichment files by table code; on duplicate tables keep the
    record with the higher self-reported confidence (tagged with its source file)."""
    merged: Dict[str, dict] = {}
    for p in paths:
        for code, rec in load_enrichment(p).items():
            if code not in merged or _conf(rec) >= _conf(merged[code]):
                merged[code] = {**rec, "_source": p.name}
    return merged


def build_rows(enriched_paths=None) -> List[dict]:
    datasets = pd.read_parquet(META / "statline_datasets.parquet").drop_duplicates("table_id")
    dims = pd.read_parquet(META / "statline_dimensions.parquet")
    meas = pd.read_parquet(META / "statline_measures.parquet")
    files = resolve_enrichment_files(enriched_paths)
    enrich = load_enrichments(files)
    print(f"[INFO] enrichment sources: {[f.name for f in files]} -> {len(enrich)} enriched tables")

    # obs_count from the full catalogue, if present.
    obs_count: Dict[str, int] = {}
    cat_path = PROC / "statline_catalog.parquet"
    if cat_path.exists():
        cat = pd.read_parquet(cat_path)
        if "ObservationCount" in cat.columns:
            obs_count = dict(zip(cat["table_id"].astype(str),
                                 pd.to_numeric(cat["ObservationCount"], errors="coerce").fillna(0).astype(int)))

    # Per-table aggregates.
    dim_text = dims.groupby("table_id")["dimension_title"].apply(_join_unique).to_dict()
    dim_count = dims.groupby("table_id")["dimension_key"].nunique().to_dict()

    def measures_blob(g: pd.DataFrame) -> str:
        parts = [_join_unique(g["title"])]
        gt = _join_unique(g["group_title"])
        if gt:
            parts.append(gt)
        descs = [d for d in g["description"].dropna().astype(str).tolist() if d.strip()][:MEASURE_DESC_CAP]
        if descs:
            parts.append(" ".join(descs))
        return " | ".join(p for p in parts if p)

    meas_text = {tid: measures_blob(g) for tid, g in meas.groupby("table_id")}
    meas_count = meas.groupby("table_id")["measure_code"].nunique().to_dict()

    rows: List[dict] = []
    for _, d in datasets.iterrows():
        tid = str(d["table_id"])
        e = enrich.get(tid, {})
        topics = e.get("topics") or []
        queries = e.get("example_queries") or []
        apps = e.get("potential_applications") or []
        rows.append({
            "table_id": tid,
            "title_nl": d.get("title") or "",
            "title_en": e.get("title_en") or "",
            "summary": (d.get("summary") or "")[:4000],
            "enriched_description": e.get("enriched_description") or "",
            "example_queries": " ".join(queries) + (" " + " ".join(apps) if apps else ""),
            "example_queries_list": "\n".join(queries),     # display (one per line)
            "applications_list": "\n".join(apps),           # display (one per line)
            "topics": ", ".join(topics) if isinstance(topics, list) else str(topics),
            "dimensions_text": dim_text.get(tid, ""),
            "measures_text": meas_text.get(tid, ""),
            "status": d.get("status") or "",
            "modified_at": str(d.get("modified_at") or ""),
            "obs_count": int(obs_count.get(tid, 0)),
            "n_dimensions": int(dim_count.get(tid, 0)),
            "n_measures": int(meas_count.get(tid, 0)),
            "source_url": d.get("source_url") or f"https://opendata.cbs.nl/#/CBS/nl/dataset/{tid}",
            "odata_url": f"{CBS_BASE}/{tid}",
            "has_enrichment": 1 if e else 0,
        })
    return rows


SCHEMA = """
DROP TABLE IF EXISTS tables;
DROP TABLE IF EXISTS tables_fts;
CREATE TABLE tables (
    rowid INTEGER PRIMARY KEY,
    table_id TEXT UNIQUE,
    title_nl TEXT, title_en TEXT, summary TEXT, enriched_description TEXT,
    example_queries TEXT, topics TEXT, dimensions_text TEXT, measures_text TEXT,
    example_queries_list TEXT, applications_list TEXT,
    status TEXT, modified_at TEXT, obs_count INTEGER,
    n_dimensions INTEGER, n_measures INTEGER,
    source_url TEXT, odata_url TEXT, has_enrichment INTEGER
);
"""

FTS_DDL = f"""
CREATE VIRTUAL TABLE tables_fts USING fts5(
    {', '.join(FTS_COLS)},
    content='tables', content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);
"""


def build_db(db_path: Path = DB_PATH, enriched_paths=None) -> int:
    rows = build_rows(enriched_paths)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.execute(FTS_DDL)
        cols = ["table_id", *FTS_COLS, "example_queries_list", "applications_list",
                "status", "modified_at", "obs_count",
                "n_dimensions", "n_measures", "source_url", "odata_url", "has_enrichment"]
        placeholders = ", ".join("?" for _ in cols)
        conn.executemany(
            f"INSERT INTO tables ({', '.join(cols)}) VALUES ({placeholders})",
            [[r[c] for c in cols] for r in rows],
        )
        # Populate external-content FTS from the base table.
        conn.execute(
            f"INSERT INTO tables_fts (rowid, {', '.join(FTS_COLS)}) "
            f"SELECT rowid, {', '.join(FTS_COLS)} FROM tables"
        )
        conn.commit()
    finally:
        conn.close()
    return len(rows)


def selftest(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    total = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    enr = conn.execute("SELECT COUNT(*) FROM tables WHERE has_enrichment=1").fetchone()[0]
    print(f"[SELFTEST] indexed tables: {total} | with enrichment: {enr}")
    for q in ["inkomen OR income", "vakanties", "werkgelegenheid OR employment", "population OR bevolking"]:
        hits = conn.execute(
            "SELECT t.table_id, t.title_nl, bm25(tables_fts) AS score "
            "FROM tables_fts JOIN tables t ON t.rowid = tables_fts.rowid "
            "WHERE tables_fts MATCH ? ORDER BY score LIMIT 3",
            (q,),
        ).fetchall()
        print(f"\n  MATCH '{q}' -> {len(hits)} top hits")
        for h in hits:
            print(f"    {h['table_id']:>10}  score={h['score']:.2f}  {h['title_nl'][:60]}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CBS metadata FTS5 search index")
    ap.add_argument("--db", default=str(DB_PATH), type=Path)
    ap.add_argument("--enriched", nargs="*", default=None,
                    help="Enrichment JSONL file(s) to index. Default: merge all "
                         "data/processed/cbs_enriched_*.jsonl (highest confidence wins on dupes).")
    ap.add_argument("--selftest", action="store_true", help="Run sanity queries after build")
    args = ap.parse_args()

    n = build_db(args.db, enriched_paths=args.enriched)
    print(f"[OK] built {args.db} with {n} tables indexed")
    if args.selftest:
        selftest(args.db)


if __name__ == "__main__":
    main()
