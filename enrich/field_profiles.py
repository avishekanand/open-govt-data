#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact per-field profiles for every table: ranges, coverage, cardinalities.

The enrichment records a 12-value SAMPLE per dimension, which is enough to ground
a prompt but not to answer "which years does this cover?", "which countries?",
"how many regions?". Those are exactly the constraints a question has to respect,
and getting them wrong is what made the period filter reject 618 usable questions.

This pass computes them exactly from the harvested code lists - no model, no
guessing:
  * period      : first, last, count, granularity (annual/quarterly/monthly)
  * geography   : count, whether EU aggregates / countries / sub-national appear
  * each field  : cardinality, first and last values, a spread sample
  * measures    : names and units
  * value range : min/max, only where observations are cached locally

    python -m enrich.field_profiles --publisher CBS
    python -m enrich.field_profiles --publisher ESTAT
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
EST_META = Path("data/processed/eurostat_metadata.jsonl")
CBS_CODES = Path("data/processed/cbs_codelists.jsonl")
EST_CATALOG = Path("data/processed/eurostat_catalog.parquet")
TABLES = Path("data/processed/tables")
OUT_TMPL = "data/processed/field_profiles_{pub}.jsonl"

YEAR = re.compile(r"(1[89]\d{2}|20\d{2})")
QUARTER = re.compile(r"\b(19|20)\d{2}[-\s]?Q[1-4]\b|kwartaal", re.I)
MONTH = re.compile(r"\b(19|20)\d{2}M\d{2}\b|maand", re.I)
WEEK = re.compile(r"\bweek\b", re.I)
# EU-level aggregates that are not countries
EU_AGG = re.compile(r"^(EU|EA|EEA|EFTA)[0-9_]*$|European Union|Euro area", re.I)


def spread(values: List[str], k: int = 12) -> List[str]:
    n = len(values)
    if n <= k:
        return list(values)
    step = (n - 1) / (k - 1)
    seen, out = set(), []
    for i in range(k):
        j = min(n - 1, round(i * step))
        if j not in seen:
            seen.add(j)
            out.append(values[j])
    return out


def period_profile(values: List[str]) -> Dict[str, Any]:
    years = sorted({int(m.group(1)) for v in values for m in [YEAR.search(str(v))] if m})
    gran = "annual"
    joined = " ".join(map(str, values[:50]))
    if WEEK.search(joined):
        gran = "weekly"
    elif MONTH.search(joined):
        gran = "monthly"
    elif QUARTER.search(joined):
        gran = "quarterly"
    return {"first_year": years[0] if years else None,
            "last_year": years[-1] if years else None,
            "n_periods": len(values), "n_years": len(years),
            "granularity": gran,
            "gaps": [y for y in range(years[0], years[-1]) if y not in set(years)][:10]
            if len(years) > 1 else []}


def geo_profile(values: List[str]) -> Dict[str, Any]:
    aggs = [v for v in values if EU_AGG.search(str(v))]
    return {"n_areas": len(values), "n_eu_aggregates": len(aggs),
            "has_subnational": any(len(str(v)) > 2 and str(v)[:2].isalpha()
                                   and any(ch.isdigit() for ch in str(v)) for v in values),
            "examples": spread(values, 10)}


def load_fields(pub: str) -> Dict[str, Dict[str, List[str]]]:
    """code -> {dimension: [category values]} from the harvested code lists."""
    out: Dict[str, Dict[str, List[str]]] = {}
    if pub == "ESTAT" and EST_META.exists():
        with EST_META.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[r["code"].lower()] = {
                    d: list((dd.get("categories") or {}).values()) or list((dd.get("categories") or {}).keys())
                    for d, dd in (r.get("dimensions") or {}).items()}
    if pub == "CBS" and CBS_CODES.exists():
        with CBS_CODES.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                out[str(r["table_id"])] = {d: (dd.get("categories") or [])
                                           for d, dd in (r.get("dimensions") or {}).items()}
    return out


def value_range(code: str) -> Optional[Dict[str, float]]:
    p = TABLES / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(p, columns=["value"])
        v = pd.to_numeric(df["value"], errors="coerce").dropna()
        return {"min": float(v.min()), "max": float(v.max()), "n": int(len(v))} if len(v) else None
    except Exception:  # noqa: BLE001
        return None


def estat_coverage() -> Dict[str, Dict[str, Any]]:
    """True Eurostat coverage from the catalogue.

    The `time` code list harvested per dataset is deliberately truncated - the
    metadata ingest fetched a single period to keep the download tiny - so it
    reports one year. Coverage must come from the catalogue instead, or every
    Eurostat period constraint is wrong.
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not EST_CATALOG.exists():
        return out
    import pandas as pd
    cat = pd.read_parquet(EST_CATALOG)
    for r in cat.itertuples():
        ds, de = str(getattr(r, "data_start", "") or ""), str(getattr(r, "data_end", "") or "")
        a, b = YEAR.search(ds), YEAR.search(de)
        if a and b:
            out[str(r.code).lower()] = {
                "first_year": int(a.group(1)), "last_year": int(b.group(1)),
                "n_years": int(b.group(1)) - int(a.group(1)) + 1,
                "granularity": ("quarterly" if "Q" in ds.upper() else
                                "monthly" if "M" in ds.upper() and ds[-2:].isdigit() else "annual"),
                "source": "catalogue"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Exact per-field profiles")
    ap.add_argument("--publisher", choices=["CBS", "ESTAT"], required=True)
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    args.out = args.out or Path(OUT_TMPL.format(pub=args.publisher.lower()))

    fields = load_fields(args.publisher)
    est_cov = estat_coverage() if args.publisher == "ESTAT" else {}
    print(f"[INFO] code lists loaded for {len(fields):,} {args.publisher} tables")

    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as fh:
        for line in src:
            r = json.loads(line)
            if r.get("publisher") != args.publisher:
                continue
            code = r["code"]
            dims = fields.get(code) or fields.get(code.lower()) or {}
            prof: Dict[str, Any] = {"code": code, "publisher": args.publisher,
                                    "title_en": r.get("title_en"), "fields": {}}
            for dim, vals in dims.items():
                vals = [str(v) for v in vals if str(v).strip()]
                if not vals:
                    continue
                entry: Dict[str, Any] = {"n_categories": len(vals),
                                         "first": vals[0], "last": vals[-1],
                                         "sample": spread(vals)}
                low = dim.lower()
                if low in ("time", "perioden") or low.startswith("perioden") or low == "period":
                    entry["period"] = period_profile(vals)
                    # Eurostat: the harvested time list is a single slice; trust the catalogue.
                    cov = est_cov.get(code.lower())
                    prof["period"] = cov if cov else entry["period"]
                    if cov:
                        entry["period_note"] = ("code list truncated by the metadata harvest; "
                                                "coverage taken from the catalogue")
                if low in ("geo", "regios", "wijkenenbuurten") or "regio" in low or low == "geo":
                    entry["geography"] = geo_profile(vals)
                    prof["geography"] = entry["geography"]
                prof["fields"][dim] = entry
            prof["measures"] = [{"name": m.get("name"), "unit": m.get("unit")}
                                for m in (r.get("measures") or [])][:20]
            vr = value_range(code)
            if vr:
                prof["value_range"] = vr
            fh.write(json.dumps(prof, ensure_ascii=False) + "\n")
            n += 1
    print(f"[DONE] {n:,} profiles -> {args.out}")


if __name__ == "__main__":
    main()
