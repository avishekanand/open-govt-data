#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance tests for the PhonePe Pulse transform (brief section 4).

The two load-bearing checks are the hard-coded ones: 26,616 district-transaction
rows and 38,664,285,691 transaction counts in 2026 Q2. Both were measured
independently of this code, so they catch a flattener that silently drops or
double-counts entries - the failure mode a row count alone would hide.

Check 3 reconciles the hierarchy across TWO views: district counts come from
map/hover, the national total from aggregated/transaction. Agreement means the
two shapes were parsed consistently, not merely parsed.

    python -m india.validate_pulse            # exit 1 if any check fails
    python -m india.validate_pulse --tol 1e-6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path("data/processed/india")
CACHE = Path("data/raw/india/pulse/data")

EXPECT_DISTRICT_ROWS = 26_616
EXPECT_DISTRICTS = 783
EXPECT_STATES = 36
EXPECT_FROM = (2018, 1)
EXPECT_TO = (2026, 2)
EXPECT_2026Q2_COUNT = 38_664_285_691


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description="Validate the India Pulse transform")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="relative tolerance for amount reconciliation (default 0.5%%)")
    args = ap.parse_args()

    txn = pd.read_parquet(OUT / "fact_transactions.parquet")
    geo = pd.read_parquet(OUT / "dim_geo.parquet")
    t = txn.join(geo.set_index("geo_id"), on="geo_id")
    dist = t[(t.level == "district") & (t.category == "TOTAL")]
    results = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print("### 1. district-transaction reconciliation")
    # A district is a (state, district) pair, never a bare name: three names are
    # shared across two states each - bilaspur (chhattisgarh / himachal pradesh),
    # hamirpur (himachal pradesh / uttar pradesh), pratapgarh (rajasthan / uttar
    # pradesh). Counting names alone gives 780 and looks like 3 missing districts.
    n_rows, n_d = len(dist), dist.geo_id.nunique()
    n_s = dist.state_name.nunique()
    periods = sorted({(int(y), int(q)) for y, q in zip(dist.year, dist.quarter)})
    check("district rows", n_rows == EXPECT_DISTRICT_ROWS,
          f"{n_rows:,} (expected {EXPECT_DISTRICT_ROWS:,})")
    check("distinct districts", n_d == EXPECT_DISTRICTS,
          f"{n_d} (expected {EXPECT_DISTRICTS})")
    check("distinct states/UTs", n_s == EXPECT_STATES,
          f"{n_s} (expected {EXPECT_STATES})")
    check("period span", periods[0] == EXPECT_FROM and periods[-1] == EXPECT_TO,
          f"{periods[0][0]}Q{periods[0][1]}-{periods[-1][0]}Q{periods[-1][1]} "
          f"({len(periods)} quarters)")

    print("\n### 2. 2026 Q2 sum check, against the national aggregate")
    q2 = dist[(dist.year == 2026) & (dist.quarter == 2)]
    got = int(q2.txn_count.sum())
    check("district sum 2026Q2", got == EXPECT_2026Q2_COUNT,
          f"{got:,} (expected {EXPECT_2026Q2_COUNT:,})")

    nat_file = CACHE / "aggregated/transaction/country/india/2026/2.json"
    nat_cats = {}
    if nat_file.exists():
        for e in json.loads(nat_file.read_text(encoding="utf-8"))["data"]["transactionData"]:
            nat_cats[e["name"]] = sum(pi.get("count") or 0
                                      for pi in (e.get("paymentInstruments") or []))
    nat_total = sum(nat_cats.values())
    check("equals national aggregate", nat_total == got,
          f"national {nat_total:,} vs districts {got:,}"
          + (f"  [{' + '.join(f'{k} {v:,}' for k, v in sorted(nat_cats.items()))}]"
             if nat_cats else ""))

    print("\n### 3. hierarchy consistency per quarter (districts vs states vs national)")
    state_tot = t[(t.level == "state") & (t.category == "TOTAL")]
    nat = t[(t.level == "country") & (t.category != "TOTAL")] \
        .groupby(["year", "quarter"], as_index=False).txn_count.sum()
    dsum = dist.groupby(["year", "quarter"], as_index=False).txn_count.sum() \
        .rename(columns={"txn_count": "districts"})
    ssum = state_tot.groupby(["year", "quarter"], as_index=False).txn_count.sum() \
        .rename(columns={"txn_count": "states"})
    h = nat.rename(columns={"txn_count": "national"}).merge(dsum, how="outer") \
        .merge(ssum, how="outer").sort_values(["year", "quarter"])
    bad = []
    for r in h.itertuples():
        for label, v in (("districts", r.districts), ("states", r.states)):
            if r.national and v and abs(v - r.national) / r.national > args.tol:
                bad.append(f"{r.year}Q{r.quarter} {label} off by "
                           f"{100*(v-r.national)/r.national:+.2f}%")
    check("sum(districts) == sum(states) == national", not bad,
          f"{len(h)} quarters within {args.tol:.1%}" if not bad
          else f"{len(bad)} mismatches: " + "; ".join(bad[:5]))

    print("\n### 4. no silent drops")
    meta = json.loads((OUT / "source_meta.json").read_text(encoding="utf-8"))
    on_disk = sum(1 for _ in CACHE.rglob("*.json"))
    check("files seen == files on disk", meta["files_seen"] == on_disk,
          f"seen {meta['files_seen']:,} | on disk {on_disk:,}")
    check("parsed + skipped == seen",
          meta["files_parsed"] + meta["files_skipped"] == meta["files_seen"],
          f"parsed {meta['files_parsed']:,} + skipped {meta['files_skipped']:,} "
          f"= {meta['files_parsed'] + meta['files_skipped']:,}")
    check("every skip has a reason",
          meta["files_skipped"] == sum(meta["skip_reasons"].values()),
          f"{meta['files_skipped']} skipped, {len(meta['skip_reasons'])} distinct reasons")

    print("\n### 5. outputs present and non-empty")
    expected = ["fact_transactions.parquet", "fact_users.parquet", "fact_merchants.parquet",
                "fact_top.parquet", "dim_geo.parquet", "dim_time.parquet",
                "dim_category.parquet", "source_meta.json",
                "csv/transactions_country.csv", "csv/transactions_state.csv",
                "csv/transactions_district.csv", "csv/users_district.csv",
                "csv/merchants_district.csv", "csv/top_states.csv", "csv/top_districts.csv"]
    missing = [f for f in expected if not (OUT / f).exists() or (OUT / f).stat().st_size == 0]
    check("all expected outputs written", not missing,
          f"{len(expected)} files" if not missing else f"missing/empty: {missing}")

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{'='*64}\n{len(results)-len(failed)}/{len(results)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        sys.exit(1)
    print("ALL CHECKS PASS")


if __name__ == "__main__":
    main()
