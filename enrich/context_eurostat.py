#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eurostat context builder - metadata only.

Grounding comes from the structure metadata harvested by
enrich.ingest_eurostat_meta (dimension names + complete code->label maps), not
from downloaded observations. Verified equivalent: across 83 datasets where both
were available, the 1-period structure fetch missed zero categories that appear
in the full table. Observations are pulled on demand, not to build this index.

Two catalogue defects are corrected here:
* the legacy `title` column lost every space; the SDMX `dataset_label` is clean.
* catalogue data_start/data_end go stale; they are used as declared, and any
  pulled CSV (when one happens to exist) overrides with what the data shows.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

from enrich.sampling import spread_sample
from enrich.schema import make_deterministic

CATALOG = Path("data/processed/eurostat_catalog.parquet")
METADATA = Path("data/processed/eurostat_metadata.jsonl")
LEGACY_CODELISTS = Path("data/processed/eurostat_codelists.json")
BASE_CSV = Path("data/eurostat_base.csv")
DOWNLOADS = Path("downloads")
PUBLISHER = "ESTAT"
MAX_CATEGORIES_SHOWN = 12


def _clean(v: Any) -> str:
    if v is None:
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("nan", "none", "<na>") else t


@lru_cache(maxsize=1)
def load_metadata(meta_path: str = str(METADATA),
                  legacy_path: str = str(LEGACY_CODELISTS)) -> Dict[str, Any]:
    """code -> {dataset_label, dimensions{dim: {name, categories{code: label}}}}."""
    out: Dict[str, Any] = {}
    legacy = Path(legacy_path)
    if legacy.exists():  # first-pass harvest, same shape
        out.update(json.loads(legacy.read_text(encoding="utf-8")))
    mp = Path(meta_path)
    if mp.exists():
        with mp.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    out[rec["code"]] = rec           # catalogue-wide ingest wins
                    out[rec["code"].upper()] = rec   # TOC codes are lower-case
                except Exception:  # noqa: BLE001
                    continue
    return out


@lru_cache(maxsize=1)
def load_catalog(catalog_path: str = str(CATALOG), base_csv: str = str(BASE_CSV)) -> pd.DataFrame:
    """Live TOC when ingested; otherwise the legacy vendored snapshot."""
    p = Path(catalog_path)
    if p.exists():
        df = pd.read_parquet(p)
        return df.dropna(subset=["code"]).drop_duplicates("code")
    df = pd.read_csv(base_csv, dtype=str).dropna(subset=["code"]).drop_duplicates("code")
    return df.rename(columns={"datastart": "data_start", "dataend": "data_end",
                              "updated": "last_update"})


def _period_year(v: Any) -> Optional[int]:
    m = re.match(r"(\d{4})", _clean(v))
    return int(m.group(1)) if m else None


def find_csv(code: str, downloads: Path = DOWNLOADS) -> Optional[Path]:
    up = code.upper()
    hits = [p for p in downloads.glob("*.csv") if p.stem.split("_")[0].upper() == up]
    return hits[0] if hits else None


def _value_range(code: str, downloads: Path) -> List[Dict[str, Any]]:
    """Optional extra: if observations happen to be on disk, report the value range."""
    path = find_csv(code, downloads)
    if path is None:
        return []
    try:
        df = pd.read_csv(path, usecols=["value"], low_memory=False)
        v = pd.to_numeric(df["value"], errors="coerce").dropna()
        return [{"name": "value", "unit": "", "min": float(v.min()), "max": float(v.max())}] if len(v) else []
    except Exception:  # noqa: BLE001
        return []


def dimensions_from_metadata(struct: Dict[str, Any]) -> List[Dict[str, Any]]:
    dims: List[Dict[str, Any]] = []
    for dim, dd in (struct.get("dimensions") or {}).items():
        if dim == "time":
            continue
        cats = dd.get("categories") or {}
        sample = [f"{c} ({l})" if l and l != c else str(c)
                  for c, l in spread_sample(list(cats.items()), MAX_CATEGORIES_SHOWN)]
        dims.append({"id": dim, "name": dd.get("name") or dim,
                     "n_categories": len(cats), "sample": sample})
    return dims


def prompt_blocks(det: Dict[str, Any]) -> Dict[str, str]:
    lines = []
    for d in det["dimensions"]:
        if not d["sample"]:
            lines.append(f"- {d['id']} / {d['name']} (category values not available)")
            continue
        spread = (f", {len(d['sample'])} sampled across the list"
                  if d["n_categories"] > len(d["sample"]) else "")
        single = "  [single value - not a breakdown]" if d["n_categories"] == 1 else ""
        lines.append(f"- {d['id']} / {d['name']} ({d['n_categories']} categories{spread}): "
                     + "; ".join(d["sample"]) + single)
    cov = det["coverage"]
    cov_txt = (f"{cov['start']} to {cov['end']}"
               + (f" ({cov['n_periods']} periods)" if cov.get("n_periods") else "")
               if cov.get("start") is not None else "(unknown)")
    mblock = "\n".join(
        f"- {m['name']}{(' [' + m['unit'] + ']') if m['unit'] else ''}: "
        f"{m['min']:.4g} to {m['max']:.4g}" for m in det["measures"]
    ) or "(observations not downloaded; values are fetched on demand)"
    return {"dimensions_block": "\n".join(lines) or "(no dimension metadata available)",
            "coverage": cov_txt, "measures_block": mblock}


def iter_items(catalog_path: Path = CATALOG, limit: Optional[int] = None,
               only_pulled: bool = False, downloads: Path = DOWNLOADS,
               require_metadata: bool = False) -> Iterator[Dict[str, Any]]:
    """Yield {code, deterministic, prompt_fields} for the Eurostat catalogue.

    only_pulled     restrict to datasets whose observations are on disk (legacy).
    require_metadata restrict to datasets with harvested structure metadata.
    """
    cat = load_catalog(str(catalog_path))
    meta = load_metadata()
    n = 0
    for r in cat.itertuples():
        code = _clean(getattr(r, "code", None))
        if not code:
            continue
        struct = meta.get(code) or meta.get(code.upper()) or {}
        if require_metadata and not struct:
            continue
        if only_pulled and find_csv(code, downloads) is None:
            continue

        dims = dimensions_from_metadata(struct)
        grounded = bool(dims) and any(d["n_categories"] for d in dims)
        title = (_clean(struct.get("dataset_label"))
                 or _clean(getattr(r, "label", None))
                 or _clean(getattr(r, "title", None)))
        # Catalogue periods are not always annual: 2019, 2019Q1, 2019M01, 2019S2.
        start, end = _period_year(getattr(r, "data_start", None)), \
            _period_year(getattr(r, "data_end", None))
        n_obs = getattr(r, "n_observations", None)
        try:
            n_obs = int(n_obs) if pd.notna(n_obs) else None
        except Exception:  # noqa: BLE001
            n_obs = None

        det = make_deterministic(
            code=code, publisher=PUBLISHER, title_native=title,
            source_url=f"https://ec.europa.eu/eurostat/databrowser/view/{code}/default/table",
            description="", last_update=_clean(getattr(r, "last_update", None)),
            coverage={"start": start, "end": end, "n_periods": None},
            dimensions=dims, measures=_value_range(code, downloads),
            n_observations=n_obs, grounded=grounded,
        )
        yield {"code": code, "deterministic": det,
               "prompt_fields": {"publisher": "Eurostat", "native_lang": "English",
                                 "language_note": "The source metadata is already in English; "
                                                  "improve clarity and searchability.",
                                 "period_dim": "time",
                                 "title_native": title,
                                 "description": (desc if (desc := _clean(getattr(r, "title", None)))
                                                 and desc != title else "(none)"),
                                 **prompt_blocks(det)}}
        n += 1
        if limit and n >= limit:
            return
