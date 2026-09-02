#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CBS StatLine context builder, emitting the unified item shape.

Same contract as enrich.context_eurostat.iter_items: metadata from the ingested
semantic layer (catalog_meta/*.parquet), real category values from the pulled
observation tables (data/processed/tables/*.parquet).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from functools import lru_cache

from enrich.sampling import spread_sample
from enrich.schema import make_deterministic

META_DIR = Path("data/processed/catalog_meta")
CODELISTS = Path("data/processed/cbs_codelists.jsonl")
TABLES_DIR = Path("data/processed/tables")
PUBLISHER = "CBS"
MAX_CATEGORIES_SHOWN = 12
MAX_MEASURES_SHOWN = 10


def _clean(v: Any) -> str:
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


@lru_cache(maxsize=1)
def load_codelists(path: str = str(CODELISTS)) -> Dict[str, Any]:
    """table_id -> {dim: {name, n_categories, categories}} harvested from OData."""
    out: Dict[str, Any] = {}
    p = Path(path)
    if p.exists():
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if rec.get("dimensions"):
                        out[str(rec["table_id"])] = rec["dimensions"]
                except Exception:  # noqa: BLE001
                    continue
    return out


def ground_from_codelists(table_id: str) -> Optional[Dict[str, Any]]:
    """Category values from the OData code lists - no observations required.

    This is the CBS counterpart to the Eurostat structure harvest. Without it,
    only tables whose observations happened to be downloaded could be grounded.
    """
    dims_map = load_codelists().get(table_id)
    if not dims_map:
        return None
    dims, coverage = [], {"start": None, "end": None, "n_periods": None}
    for dim, dd in dims_map.items():
        cats = dd.get("categories") or []
        if not cats:
            continue
        n = dd.get("n_categories") or len(cats)
        if dim.lower().startswith("perioden"):
            years = pd.to_numeric(pd.Series(cats).str.extract(r"(\d{4})")[0],
                                  errors="coerce").dropna().astype(int)
            if len(years):
                coverage = {"start": int(years.min()), "end": int(years.max()),
                            "n_periods": n}
        dims.append({"id": dim.strip(), "name": (dd.get("name") or dim).strip(), "n_categories": n,
                     "sample": spread_sample(cats, MAX_CATEGORIES_SHOWN)})
    if not dims:
        return None
    return {"dimensions": dims, "coverage": coverage, "measures": [],
            "n_observations": None}


def ground_from_parquet(table_id: str, tables_dir: Path = TABLES_DIR) -> Optional[Dict[str, Any]]:
    path = tables_dir / f"{table_id}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return None

    dims: List[Dict[str, Any]] = []
    coverage = {"start": None, "end": None, "n_periods": None}
    for col in [c for c in df.columns if c.endswith("_label")]:
        dim = col[: -len("_label")]
        vals = [str(v).strip() for v in pd.unique(df[col].dropna()) if str(v).strip()]
        if not vals:
            continue
        if dim.lower().startswith("perioden"):
            years = pd.to_numeric(pd.Series(vals).str.extract(r"(\d{4})")[0],
                                  errors="coerce").dropna().astype(int)
            if len(years):
                coverage = {"start": int(years.min()), "end": int(years.max()),
                            "n_periods": len(vals)}
        dims.append({"id": dim.strip(), "name": dim.strip(), "n_categories": len(vals),
                     "sample": spread_sample(vals, MAX_CATEGORIES_SHOWN)})

    measures: List[Dict[str, Any]] = []
    if {"measure", "value"} <= set(df.columns):
        num = df.dropna(subset=["value"])
        for name, grp in list(num.groupby("measure", sort=False))[:MAX_MEASURES_SHOWN]:
            unit = ""
            if "unit" in grp.columns:
                u = grp["unit"].dropna()
                unit = _clean(u.iloc[0]) if len(u) else ""
            measures.append({"name": str(name), "unit": unit,
                             "min": float(grp["value"].min()), "max": float(grp["value"].max())})
    return {"dimensions": dims, "coverage": coverage, "measures": measures,
            "n_observations": int(len(df))}


def prompt_blocks(det: Dict[str, Any], meta_measures: List[str]) -> Dict[str, str]:
    lines = []
    for d in det["dimensions"]:
        if not d["sample"]:  # metadata-only fallback: no observations pulled
            lines.append(f"- {d['id']} (category values not available)")
            continue
        spread = (f", {len(d['sample'])} sampled across the list"
                  if d["n_categories"] > len(d["sample"]) else "")
        single = "  [single value - not a breakdown]" if d["n_categories"] == 1 else ""
        lines.append(f"- {d['id']} ({d['n_categories']} categories{spread}): "
                     + "; ".join(d["sample"]) + single)
    cov = det["coverage"]
    cov_txt = (f"{cov['start']} to {cov['end']} ({cov['n_periods']} periods)"
               if cov["start"] is not None else "(unknown)")
    if det["measures"]:
        mblock = "\n".join(
            f"- {m['name']}{(' [' + m['unit'] + ']') if m['unit'] else ''}: "
            f"{m['min']:.4g} to {m['max']:.4g}" for m in det["measures"])
    else:  # not pulled: fall back to the measure titles from the metadata layer
        mblock = ("- " + "; ".join(meta_measures[:25])) if meta_measures else "(none)"
    return {"dimensions_block": "\n".join(lines) or "(no dimension data pulled)",
            "coverage": cov_txt, "measures_block": mblock}


def iter_items(meta_dir: Path = META_DIR, tables_dir: Path = TABLES_DIR,
               limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    datasets = pd.read_parquet(meta_dir / "statline_datasets.parquet").drop_duplicates("table_id")
    dims_df = pd.read_parquet(meta_dir / "statline_dimensions.parquet")
    meas_df = pd.read_parquet(meta_dir / "statline_measures.parquet")
    dim_titles = dims_df.groupby("table_id")["dimension_title"].apply(
        lambda s: [x for x in s.dropna().tolist()]).to_dict()
    meas_titles = meas_df.groupby("table_id")["title"].apply(
        lambda s: [x for x in s.dropna().tolist()]).to_dict()

    n = 0
    for r in datasets.itertuples():
        tid = str(r.table_id)
        # Pulled observations are richer (real value ranges), but exist for only a
        # fraction of tables; the harvested code lists cover everything.
        g = ground_from_parquet(tid, tables_dir)
        if g is None or not g["dimensions"]:
            g = ground_from_codelists(tid) or g
        elif not g["coverage"].get("start"):
            cl = ground_from_codelists(tid)
            if cl and cl["coverage"].get("start"):
                g["coverage"] = cl["coverage"]
        grounded = g is not None and bool(g.get("dimensions"))
        if not grounded:
            # metadata-only fallback: dimension titles, no real categories
            g = {"dimensions": [{"id": t.strip(), "name": t.strip(), "n_categories": 0,
                                 "sample": []}
                                for t in dim_titles.get(tid, [])],
                 "coverage": {"start": None, "end": None, "n_periods": None},
                 "measures": [], "n_observations": None}
        period_dim = next((d["id"] for d in g["dimensions"]
                           if d["id"].lower().startswith("perioden")), "Perioden")
        det = make_deterministic(
            code=tid, publisher=PUBLISHER, title_native=_clean(getattr(r, "title", None)),
            source_url=_clean(getattr(r, "source_url", None)),
            description=_clean(getattr(r, "summary", None))[:1500],
            last_update=_clean(getattr(r, "modified_at", None)),
            coverage=g["coverage"], dimensions=g["dimensions"], measures=g["measures"],
            n_observations=g["n_observations"], grounded=grounded,
        )
        yield {"code": tid, "deterministic": det,
               "prompt_fields": {"publisher": "Statistics Netherlands (CBS)",
                                 "native_lang": "Dutch",
                                 "language_note": "The source metadata is in Dutch; "
                                                  "write the enrichment in ENGLISH.",
                                 "period_dim": period_dim,
                                 "title_native": det["title_native"],
                                 "description": det["description_native"] or "(none)",
                                 **prompt_blocks(det, meas_titles.get(tid, []))}}
        n += 1
        if limit and n >= limit:
            return
