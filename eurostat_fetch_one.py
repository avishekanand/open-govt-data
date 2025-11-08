#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
eurostat_fetch_one_verbose.py
Download a single Eurostat dataset, print summary + post-processed tables, and save CSV.

Usage examples:
  python eurostat_fetch_one_verbose.py --code TPS00001 --out population.csv
  python eurostat_fetch_one_verbose.py --code TOUR_OCC_NINRAW --filter geo=NL time=2023 --out tourism.csv
  python eurostat_fetch_one_verbose.py --code TRNG_LFS_22 --filter freq=A unit=PC time=2023 --out training.csv --prefer-labels
"""

import argparse
import sys
import time
from typing import Dict, List, Any, Tuple, Optional

import requests
import pandas as pd

# Optional pretty tables
try:
    from tabulate import tabulate
    HAVE_TABULATE = True
except Exception:
    HAVE_TABULATE = False

API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
UA = "eurostat-fetch-one/1.4 (contact: your-email@example.com)"


# ---------------- Console helpers ----------------
def print_table(df: pd.DataFrame, title: str, max_rows: int = 15):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    if df is None or df.empty:
        print("(no rows)")
        return
    shown = df.head(max_rows)
    if HAVE_TABULATE:
        print(tabulate(shown, headers="keys", tablefmt="github", showindex=False))
    else:
        print(shown.to_string(index=False))


def print_kv(kv: Dict[str, Any], title: str):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    for k, v in kv.items():
        print(f"{k}: {v}")


# ---------------- Fetch ----------------
def fetch_eurostat_dataset(code: str, filters: Dict[str, str], timeout: int = 60, retries: int = 3) -> dict:
    url = f"{API_BASE}/{code}"
    params = filters or {}
    headers = {"User-Agent": UA, "Accept": "application/json"}
    backoff = 2.0
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries:
                raise
            print(f"[WARN] Attempt {attempt} failed: {e}. Retrying in {backoff:.1f}s...", file=sys.stderr)
            time.sleep(backoff)
            backoff *= 1.8
    return {}


# ---------------- Dimension discovery & maps ----------------
def _discover_dim_ids(dim_obj: dict) -> List[str]:
    """
    Prefer dim_obj['id']; otherwise infer keys that have category.index.
    Ensure 'time' is last (Eurostat ordering).
    """
    ids = dim_obj.get("id", [])
    if isinstance(ids, list) and ids:
        return ids
    candidates = []
    for k, v in dim_obj.items():
        if k in {"id", "size", "role"}:
            continue
        if isinstance(v, dict) and "category" in v and "index" in v["category"]:
            candidates.append(k)
    if "time" in candidates:
        candidates = [x for x in candidates if x != "time"] + ["time"]
    return candidates


def _build_dim_index_maps(dim_obj: dict) -> Tuple[List[str], Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    dim_ids = _discover_dim_ids(dim_obj)
    dim_codes_by_index, dim_labels = {}, {}
    for dim in dim_ids:
        d = dim_obj.get(dim, {})
        cat = d.get("category", {})
        idx_map = cat.get("index", {})  # code -> position
        inv = sorted(idx_map.items(), key=lambda kv: kv[1])
        codes = [code for code, _ in inv]
        dim_codes_by_index[dim] = codes
        labels = cat.get("label", {}) or d.get("label", {})
        if isinstance(labels, dict) and labels:
            dim_labels[dim] = labels
    return dim_ids, dim_codes_by_index, dim_labels


# ---------------- Flatten ----------------
def flatten_sdmx_json(data: dict, prefer_labels: bool = False) -> pd.DataFrame:
    """
    Flatten a Eurostat dataset response. Supports:
      - JSON-stat 2.0 format (value keys = single integer index)
      - SDMX-JSON 1.x format (value keys = colon-separated indices)
    Returns a pandas DataFrame with one row per observation and dimension columns.
    """
    if not data or "dimension" not in data:
        raise ValueError("Unexpected JSON structure: missing 'dimension'")

    dim_obj = data["dimension"]
    # Build dimension IDs, codes, labels
    def _discover_ids(dim_obj):
        ids = dim_obj.get("id", [])
        if isinstance(ids, list) and ids:
            return ids
        # fallback: infer keys having category.index
        candidates = [k for k, v in dim_obj.items()
                      if isinstance(v, dict) and "category" in v and "index" in v["category"]]
        if "time" in candidates:
            candidates = [c for c in candidates if c != "time"] + ["time"]
        return candidates

    dim_ids = _discover_ids(dim_obj)
    dim_codes_by_index = {}
    dim_labels = {}
    for dim in dim_ids:
        d = dim_obj.get(dim, {})
        cat = d.get("category", {})
        idx_map = cat.get("index", {})
        inv = sorted(idx_map.items(), key=lambda kv: kv[1])
        codes = [code for code, _ in inv]
        dim_codes_by_index[dim] = codes
        labels = cat.get("label", {}) or d.get("label", {})
        if isinstance(labels, dict) and labels:
            dim_labels[dim] = labels

    # Choose decoding path
    values = data.get("value", {})
    # Check JSON-stat style: numeric keys, plus size array
    if "version" in data and str(data["version"]).startswith("2") and all(k.isdigit() for k in map(str, values.keys())):
        # JSON-stat 2.0 path
        sizes = data.get("size", [])
        if len(sizes) != len(dim_ids):
            print(f"[WARN] size array length ({len(sizes)}) != number of dims ({len(dim_ids)})", file=sys.stderr)
        # compute strides (row-major flattening)
        strides = []
        cum = 1
        for s in sizes[::-1]:
            strides.insert(0, cum)
            cum *= s
        rows = []
        for key, val in values.items():
            idx = int(key)
            coords = []
            rem = idx
            for stride, size in zip(strides, sizes):
                c = rem // stride
                rem = rem % stride
                coords.append(c)
            rec = {}
            for dim, c in zip(dim_ids, coords):
                codes = dim_codes_by_index.get(dim, [])
                rec[dim] = codes[c] if 0 <= c < len(codes) else None
                if prefer_labels and dim in dim_labels and rec[dim] in dim_labels[dim]:
                    rec[f"{dim}_label"] = dim_labels[dim][rec[dim]]
            rec["value"] = val
            rows.append(rec)
        df = pd.DataFrame(rows)
    else:
        # SDMX-JSON 1.x fallback path; keys like "0:1:2"
        statuses = data.get("status", {})
        rows = []
        for key, val in values.items():
            idxs = [int(x) for x in key.split(":")] if ":" in key else [int(key)]
            if len(idxs) != len(dim_ids):
                print(f"[WARN] Observation key indices ({len(idxs)}) ≠ dimensions ({len(dim_ids)}). Proceeding best-effort.", file=sys.stderr)
            rec = {}
            for dim, idx in zip(dim_ids, idxs):
                codes = dim_codes_by_index.get(dim, [])
                rec[dim] = codes[idx] if 0 <= idx < len(codes) else None
                if prefer_labels and dim in dim_labels and rec[dim] in dim_labels[dim]:
                    rec[f"{dim}_label"] = dim_labels[dim][rec[dim]]
            rec["value"] = val
            st = statuses.get(key)
            if st is not None:
                rec["status"] = st
            rows.append(rec)
        df = pd.DataFrame(rows)

    # Ensure all dims exist as columns
    for dim in dim_ids:
        if dim not in df.columns:
            df[dim] = pd.NA

    # Rearrange columns: dims, labels, value, (status), rest
    label_cols = [f"{d}_label" for d in dim_ids if prefer_labels and f"{d}_label" in df.columns]
    ordered = [c for c in (dim_ids + label_cols + ["value", "status"]) if c in df.columns]
    tail = [c for c in df.columns if c not in ordered]
    if not df.empty:
        df = df[ordered + tail]

    return df

# ---------------- Post-processing views ----------------
def dataset_summary(code: str, data: dict, df: pd.DataFrame) -> Dict[str, Any]:
    dim_obj = data.get("dimension", {}) or {}
    dims = _discover_dim_ids(dim_obj)
    title = data.get("label", "")
    updated = data.get("updated", "")
    time_dim = dim_obj.get("time", {})
    time_cov = ""
    if "category" in time_dim and "label" in time_dim["category"]:
        times = list(time_dim["category"]["label"].keys())
        if times:
            time_cov = f"{times[0]} … {times[-1]} (total {len(times)})"
    return {
        "Dataset": code,
        "URL": f"{API_BASE}/{code}",
        "Title": title,
        "Last updated": updated,
        "Dimensions": f"{len(dims)} → " + ", ".join(dims),
        "Time coverage": time_cov,
        "Observations": f"{len(df):,}",
        "Flattened columns": f"{len(df.columns)}",
    }


def value_status_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = pd.Series(dtype="object")
    s.loc["rows"] = len(df)
    vals = pd.to_numeric(df["value"], errors="coerce") if "value" in df.columns else pd.Series([], dtype="float")
    s.loc["value_non_null"] = int(vals.notna().sum())
    s.loc["value_null"] = int(vals.isna().sum())
    if "status" in df.columns:
        st = df["status"].fillna("(none)").value_counts().head(10)
        st.index = [f"status:{k}" for k in st.index]
        s = pd.concat([s, st])
    return s.to_frame(name="count").reset_index(names="metric")


def dimension_cards(df: pd.DataFrame, dims: List[str], prefer_labels: bool) -> pd.DataFrame:
    rows = []
    for d in dims:
        if d not in df.columns:
            continue
        non_na = df[d].dropna()
        uniq = non_na.astype(str).unique()
        examples = ", ".join(list(map(str, uniq[:6]))) if len(uniq) else "(all NA)"
        card = {"dimension": d, "unique_members": int(len(uniq)), "examples": examples}
        if prefer_labels and f"{d}_label" in df.columns and len(uniq):
            lab_uniq = df[[d, f"{d}_label"]].dropna().drop_duplicates().head(3)
            ex_lab = "; ".join([f"{r[d]}: {r[f'{d}_label']}" for _, r in lab_uniq.iterrows()])
            if ex_lab:
                card["examples"] = f"{card['examples']} | {ex_lab}"
        rows.append(card)
    return pd.DataFrame(rows)


def top10_latest_year(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "time" not in df.columns or df["time"].dropna().empty:
        return None
    if "geo" not in df.columns or df["geo"].dropna().empty:
        return None
    try:
        latest = sorted(df["time"].dropna().astype(str).unique())[-1]
    except Exception:
        return None
    sub = df[df["time"].astype(str) == str(latest)].copy()
    if sub.empty:
        return None
    sub["value_num"] = pd.to_numeric(sub["value"], errors="coerce")
    out = sub[["geo", "time", "value_num"]].dropna().sort_values("value_num", ascending=False).head(10)
    if out.empty:
        return None
    out.rename(columns={"value_num": "value"}, inplace=True)
    return out


def subgroup_pivot_latest(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Safely build a subgroup pivot for latest year, only if a subgroup dim has >=2 non-NA categories."""
    if "time" not in df.columns or df["time"].dropna().empty:
        return None
    try:
        latest = sorted(df["time"].dropna().astype(str).unique())[-1]
    except Exception:
        return None
    sub_all = df[df["time"].astype(str) == str(latest)].copy()
    if sub_all.empty:
        return None

    # Candidate subgroup dims ordered by usefulness
    candidates = [d for d in ["sex", "age", "reason", "rskpovth", "c_resid"] if d in sub_all.columns]

    # Pick the first with at least 2 non-NA categories
    pick = None
    for d in candidates:
        non_na = sub_all[d].dropna()
        if non_na.nunique() >= 2:
            pick = d
            break
    if pick is None:
        return None  # nothing to pivot on

    # If multiple geos, pick the geo with highest total value
    if "geo" in sub_all.columns and sub_all["geo"].dropna().nunique() > 1:
        sub_all["value_num"] = pd.to_numeric(sub_all["value"], errors="coerce")
        geo_sum = sub_all.groupby("geo", dropna=True)["value_num"].sum().sort_values(ascending=False)
        if not geo_sum.empty:
            sub_all = sub_all[sub_all["geo"] == geo_sum.index[0]]

    sub_all["value_num"] = pd.to_numeric(sub_all["value"], errors="coerce")
    pv = sub_all.pivot_table(index=pick, values="value_num", aggfunc="mean", dropna=True)
    pv = pv.reset_index().rename(columns={pick: pick, "value_num": "value"}).sort_values("value", ascending=False)
    if pv.empty:
        return None
    return pv.head(10)


# ---------------- Printing ----------------
def print_dataset_summary(code: str, data: dict, df: pd.DataFrame, prefer_labels: bool):
    print("=" * 80)
    print(f"DATASET: {code}")
    print(f"URL: {API_BASE}/{code}")

    dim_obj = data.get("dimension", {}) or {}
    dims = _discover_dim_ids(dim_obj)
    title = data.get("label")
    if title:
        print(f"→ Title: {title}")
    updated = data.get("updated")
    if updated:
        print(f"→ Last updated: {updated}")

    time_dim = dim_obj.get("time", {})
    if "category" in time_dim and "label" in time_dim["category"]:
        times = list(time_dim["category"]["label"].keys())
        if times:
            print(f"→ Time coverage: {times[0]} … {times[-1]} (total {len(times)})")

    print(f"→ Dimensions ({len(dims)}): {', '.join(dims)}")
    print(f"→ Total observations: {len(df):,}")
    print(f"→ Columns in flattened CSV: {len(df.columns)}")

    # First rows
    print_table(df.head(5), "First 5 rows (flattened)")

    # Value/status summary
    vs = value_status_summary(df)
    print_table(vs, "Value / Status summary")

    # Dimension cards
    dim_df = dimension_cards(df, dims, prefer_labels)
    print_table(dim_df, "Dimension cards (unique members + examples)")

    # Top-10 latest year by value
    top = top10_latest_year(df)
    if top is not None and not top.empty:
        print_table(top, "Top 10 by value in latest year")
    else:
        print_table(pd.DataFrame(), "Top 10 by value in latest year (no suitable geo/time/value)")

    # Subgroup pivot (latest year)
    piv = subgroup_pivot_latest(df)
    if piv is not None and not piv.empty:
        print_table(piv, "Subgroup pivot in latest year (top geography if multiple)")
    else:
        print_table(pd.DataFrame(), "Subgroup pivot (no subgroup dimension with ≥2 non-NA categories)")


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Download a Eurostat dataset, print metadata + tables, and save CSV.")
    ap.add_argument("--code", required=True, help="Dataset code, e.g., TPS00001")
    ap.add_argument("--filter", nargs="*", default=[], help="Optional filters: geo=NL time=2023 freq=A unit=PC")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--prefer-labels", action="store_true", help="Add *_label columns when available")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    # Parse filters
    filters: Dict[str, str] = {}
    for f in args.filter:
        if "=" in f:
            k, v = f.split("=", 1)
            filters[k.strip()] = v.strip()

    print(f"[INFO] Fetching {args.code} with filters={filters or '{}'}")
    data = fetch_eurostat_dataset(args.code, filters, timeout=args.timeout, retries=args.retries)
    df = flatten_sdmx_json(data, prefer_labels=args.prefer_labels)

    meta = dataset_summary(args.code, data, df)
    print_kv(meta, "Dataset summary")
    print_dataset_summary(args.code, data, df, args.prefer_labels)

    df.to_csv(args.out, index=False)
    print(f"[OK] Saved to {args.out}")


if __name__ == "__main__":
    main()