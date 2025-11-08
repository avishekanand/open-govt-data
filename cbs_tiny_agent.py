# cbs_tiny_agent.py
# Minimal NL→Plan→OData→Execute→Post-process pipeline (CBS StatLine)
# Dataset: 83131ENG  — Consumer prices; price index 2015=100
#
# What this does:
# - Uses the correct OData endpoint and field keys:
#     * Periods (time dimension)
#     * ExpenditureCategories (category dimension)
#     * CPI_1 (measure: index 2015=100)
# - Planner with back-compat wrapper (accepts "YYYY-MM" and converts to "YYYYMMnn" style 'YYYYMM##' => 'YYYYMM01')
# - Compiler generates a valid OData request (period values quoted)
# - Executor fetches all pages (with polite retry) and caches per-run
# - Post-processor computes YoY and draws two simple charts
#
# Usage:
#   pip install requests pandas matplotlib python-dateutil
#   python cbs_tiny_agent.py

import json
import time
import hashlib
import requests
import pandas as pd
import matplotlib.pyplot as plt
import requests

# ============== 0) Tiny CATALOG (exact keys from DataProperties) ==============
CATALOG = {
    "CPI_2015_100": {
        "id": "83131ENG",
        "title": "Consumer prices; price index 2015=100",
        "base_url": "https://opendata.cbs.nl/ODataApi/OData/83131ENG/TypedDataSet",
        "period_field": "Periods",                # time dimension key
        "category_field": "ExpenditureCategories",# category dimension key
        "measure": "CPI_1",                       # main CPI index (unit: 2015=100)
        "unit": "index(2015=100)",
        "codes": { "All items": "000000" },       # 'All items' category code
        "page_size": 10000
    }
}

# In-memory per-run cache (prevents duplicate HTTP requests inside one run)
CACHE: dict[str, pd.DataFrame] = {}

# ============== Helpers for period codes and provenance =======================

def to_cbs_month(code_yyyy_mm: str) -> str:
    """
    Convert 'YYYY-MM' -> CBS OData monthly code 'YYYYMM##' style.
    For CPI monthly, CBS uses 'YYYYMMnn' with 'MM' literal, e.g., '2024MM01' for Jan 2024.
    """
    yyyy, mm = code_yyyy_mm.split("-")
    return f"{yyyy}MM{mm.zfill(2)}"

def period_str_to_datetime(period_code: str) -> pd.Timestamp:
    """
    Map CBS 'Periods' strings to pandas Timestamps for sorting/lag.
      - Monthly like '2024MM01' => 2024-01-01
      - Annual like '2024JJ00'  => 2024-01-01 (coarse)
    Extend as needed for other period formats.
    """
    if "MM" in period_code:
        yyyy = period_code[:4]
        mm = period_code[-2:]
        return pd.Timestamp(f"{yyyy}-{mm}-01")
    if "JJ" in period_code:  # annual
        yyyy = period_code[:4]
        return pd.Timestamp(f"{yyyy}-01-01")
    # Fallback: try plain year-month
    try:
        return pd.to_datetime(period_code)
    except Exception:
        return pd.NaT

# ==================== 1) Minimal “Planner” (two variants) =====================

def plan_cpi_all_items(start="2024MM01", end="2025MM10") -> dict:
    """
    Minimal plan for CPI 'All items' over [start, end] (inclusive).
    Expects CBS monthly codes 'YYYYMMnn' (e.g., '2024MM01').
    """
    m = CATALOG["CPI_2015_100"]
    return {
        "dataset_key": "CPI_2015_100",
        "dataset_id": m["id"],
        "select": [m["period_field"], m["category_field"], m["measure"]],
        "filters": { m["category_field"]: [m["codes"]["All items"]] },
        "time": {"from": start, "to": end, "freq": "M"},
        "calcs": ["YoY"],
        "limit": 50000
    }

def plan_cpi(country: str = "Netherlands", start: str = "2024-01", end: str = "2025-10") -> dict:
    """
    Backward-compatible wrapper: accepts 'YYYY-MM' and converts to CBS monthly codes.
    'country' is unused for CPI (no region dim), kept for API compatibility only.
    """
    return plan_cpi_all_items(start=to_cbs_month(start), end=to_cbs_month(end))

# ========================= 2) Compiler: Plan -> OData =========================

def compile_to_odata(plan: dict) -> dict:
    m = CATALOG[plan["dataset_key"]]
    p_from = plan["time"]["from"]
    p_to   = plan["time"]["to"]
    cat    = plan["filters"][m["category_field"]][0]

    # IMPORTANT: quote Periods values (they are string codes)
    filt = (
        f"{m['category_field']} eq '{cat}' and "
        f"{m['period_field']} ge '{p_from}' and {m['period_field']} le '{p_to}'"
    )
    params = {
        "$select": ",".join(plan["select"]),
        "$filter": filt,
        "$orderby": f"{m['period_field']} asc",
        "$top": m["page_size"],
        "$skip": 0
    }
    return { "url": m["base_url"], "params": params, "dataset_id": m["id"] }

# ================== 3) Executor: paged GET + tiny in-run cache =================

def _spec_cache_key(spec: dict) -> str:
    return hashlib.md5(json.dumps({"u": spec["url"], "p": spec["params"]}, sort_keys=True).encode()).hexdigest()

def execute_all_pages(spec: dict, max_pages: int = 100) -> pd.DataFrame:
    key = _spec_cache_key(spec)
    if key in CACHE:
        return CACHE[key].copy()

    rows = []
    skip = 0
    top = int(spec["params"]["$top"])

    for _ in range(max_pages):
        page_params = dict(spec["params"], **{"$skip": skip})
        r = requests.get(spec["url"], params=page_params, timeout=30)
        if r.status_code == 429:
            time.sleep(1.0)
            r = requests.get(spec["url"], params=page_params, timeout=30)
        r.raise_for_status()
        block = r.json().get("value", [])
        rows.extend(block)
        if len(block) < top:
            break
        skip += top

    df = pd.DataFrame.from_records(rows)
    CACHE[key] = df.copy()
    return df

# ========================= 4) Post-processing helpers =========================

def compute_yoy_monthly(df: pd.DataFrame, period_col: str, value_col: str) -> pd.DataFrame:
    """
    Compute YoY % change assuming monthly frequency.
    """
    out = df.copy()
    out["_dt"] = out[period_col].astype(str).map(period_str_to_datetime)
    out = out.sort_values("_dt").reset_index(drop=True)
    out["lag_12"] = out[value_col].shift(12)
    out["YoY_pct"] = (out[value_col] / out["lag_12"] - 1.0) * 100.0
    return out

def quick_line(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str):
    plt.figure(figsize=(9,4))
    plt.plot(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.xlabel("Period")
    plt.ylabel(y_label)
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()


def fetch_codes(dataset_id: str, entity: str, select="Key,Title", order=None, top=1000):
    base = f"https://opendata.cbs.nl/ODataApi/OData/{dataset_id}/{entity}"
    params = {"$select": select, "$top": top}
    if order:
        params["$orderby"] = order
    r = requests.get(base, params=params, timeout=30); r.raise_for_status()
    return r.json().get("value", [])

def find_all_items_code(dataset_id="83131ENG") -> str:
    items = fetch_codes(dataset_id, "ExpenditureCategories", "Key,Title", top=500)
    # Try common labels; fallback to first code if unknown
    for it in items:
        title = (it.get("Title") or "").strip().lower()
        if title in {"all items", "totaal", "alle artikelen"}:
            return it["Key"]
    # Also try the classic '000000' if present
    for it in items:
        if it["Key"] == "000000":
            return "000000"
    # fallback
    return items[0]["Key"]

def latest_periods(dataset_id="83131ENG", n=24) -> list[str]:
    vals = fetch_codes(dataset_id, "Periods", "Key,Title", order="Key desc", top=n)
    return [v["Key"] for v in vals]

def plan_cpi_latest(months=24) -> dict:
    m = CATALOG["CPI_2015_100"]
    cat_code = find_all_items_code(m["id"])
    periods = latest_periods(m["id"], n=max(months, 3))
    if not periods:
        raise RuntimeError("No Periods found in dataset.")
    # Build a range [oldest..newest] within what actually exists
    p_to = periods[0]               # newest by desc
    p_from = periods[min(len(periods)-1, months-1)]  # ~months back if available
    return {
        "dataset_key": "CPI_2015_100",
        "dataset_id": m["id"],
        "select": [m["period_field"], m["category_field"], m["measure"]],
        "filters": { m["category_field"]: [cat_code] },
        "time": {"from": p_from, "to": p_to, "freq": "M"},
        "calcs": ["YoY"],
        "limit": 50000
    }

# ================================ 5) Main =====================================

if __name__ == "__main__":
    # --- A) Build a plan (either back-compat or direct CBS codes) ---
    # Option 1: back-compat "YYYY-MM"
    # plan = plan_cpi(start="2024-01", end="2025-10")

    # Option 2: direct CBS monthly codes
    # plan = plan_cpi_all_items(start="2024MM01", end="2025MM10")
    plan = plan_cpi_latest(months=24)

    # --- B) Compile to OData request ---
    spec = compile_to_odata(plan)

    print("OData request:", spec["url"])
    print("Params:", spec["params"])

    # --- C) Execute (paged) ---
    df = execute_all_pages(spec)

    # Guard: ensure measure is present (CBS sometimes renames; we use catalog key)
    meta = CATALOG[plan["dataset_key"]]
    period_col = meta["period_field"]
    value_col = meta["measure"]
    if value_col not in df.columns:
        # try to discover CPI column by fallback (shouldn't trigger given correct key)
        for col in df.columns:
            if col.upper().startswith("CPI"):
                value_col = col
                break

    if df.empty:
        raise SystemExit("No rows returned. Try widening the period or verify the dataset is available.")

    # --- D) Post-process (YoY) and show preview ---
    df2 = compute_yoy_monthly(df, period_col=period_col, value_col=value_col)
    preview_cols = [c for c in [period_col, meta["category_field"], value_col, "YoY_pct"] if c in df2.columns]
    print(df2[preview_cols].tail(8))

    # --- E) Plots ---
    quick_line(df2, x_col=period_col, y_col=value_col,
               title="CPI (2015=100) — All items — Netherlands",
               y_label=meta["unit"])
    quick_line(df2, x_col=period_col, y_col="YoY_pct",
               title="YoY CPI (%) — All items — Netherlands",
               y_label="YoY (%)")

    # --- F) Simple provenance ---
    print("\nProvenance:")
    print(f"- Dataset: {meta['id']} — {meta['title']}")
    print(f"- API: {meta['base_url']}")
    print(f"- Fields: Periods='{period_col}', Category='{meta['category_field']}', Measure='{value_col}'")
