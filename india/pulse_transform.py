#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flatten the PhonePe Pulse mirror into a parquet star schema + tidy CSVs.

Three JSON shapes, not one, and the shape depends on BOTH the view and the
metric:

  aggregated/transaction   data.transactionData[]  -> {name, paymentInstruments[]}
  aggregated/user|merchant data.aggregated         -> {registeredCount}
  map/transaction          data.hoverDataList[]    -> {name, metric[]}      (a LIST)
  map/user|merchant        data.hoverData          -> {name: {...}}         (a DICT)
  top/transaction          data.{states,districts} -> {entityName, metric}
  top/user|merchant        data.{states,districts} -> {name, registeredCount}

Everything is read with .get and optional fields are nullable, because the repo
has been restructured and the field set is not uniform across years. What the
mirror at commit 257bf38 actually contains, measured over all 11,184 files
rather than assumed:

  * `amount` NEVER appears in aggregated/transaction (0 of 3,773 entries).
    Rupee amounts exist ONLY in the map/hover view. So fact_transactions carries
    counts with a null amount for country/state category rows, and count+amount
    for the TOTAL rows that come from map.
  * `usersByDevice` and `appOpens` appear in ZERO files. The brief expected both
    in older years; the restructure removed them. fact_users_by_device therefore
    has no source and is not written (writing an empty table would imply the
    measure exists and is zero). app_opens is kept as an all-null column.
  * transaction categories are only P2P, Retail, Utility across every year - the
    older "Recharge & bill payments" / "Financial Services" buckets are gone.
    The raw `name` is still stored verbatim so a future category is not lost.
  * top/<metric>/country/india/state/... always has `states: null` (a state file
    ranks its own districts). Null, not missing - hence `or []` everywhere.

    python -m india.pulse_transform --dry-run     # parse + report, write nothing
    python -m india.pulse_transform               # -> data/processed/india/
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CACHE = Path("data/raw/india/pulse/data")
OUT = Path("data/processed/india")
CSV = OUT / "csv"
META_CACHE = Path("data/raw/india/_commit_main.json")

SOURCE_URL = "https://github.com/PhonePe/pulse"
LICENCE = "CDLA-Permissive-2.0"
LICENCE_URL = "https://cdla.dev/permissive-2-0/"
ATTRIBUTION = "PhonePe Pulse (https://www.phonepe.com/pulse/), CDLA-Permissive-2.0"

CATEGORY_BUCKET = {
    "p2p": "peer_to_peer",
    "retail": "retail_merchant",
    "utility": "utility_and_bills",
    "total": "all_categories",
}
QUARTER_SPAN = {1: ("01-01", "03-31"), 2: ("04-01", "06-30"),
                3: ("07-01", "09-30"), 4: ("10-01", "12-31")}


# ---------------------------------------------------------------- helpers

def norm_district(name: str) -> str:
    """`Angul district` -> `angul`. The suffix is inconsistent in the source."""
    s = re.sub(r"\s+", " ", (name or "").strip().lower())
    return re.sub(r"\s+district$", "", s).strip()


def slug_of(state_name: str) -> str:
    """`andaman & nicobar islands` -> `andaman-&-nicobar-islands` (the repo's slug)."""
    return re.sub(r"\s+", "-", (state_name or "").strip().lower())


def label_of(slug: str) -> str:
    return slug.replace("-", " ").strip()


def geo_country() -> str:
    return "IN"


def geo_state(slug: str) -> str:
    return f"IN-{slug}"


def geo_district(slug: str, district: str) -> str:
    return f"IN-{slug}-{district}"


def _level(path_parts: Tuple[str, ...], rel: str) -> str:
    if "/hover/" in rel:
        return "district" if "/state/" in rel.split("/hover/")[1] else "state-entries"
    return "state" if "/state/" in rel else "country"


def _state_slug_from(rel: str) -> Optional[str]:
    m = re.search(r"/state/([^/]+)/", rel)
    return m.group(1) if m else None


# ---------------------------------------------------------------- parse

class Collector:
    def __init__(self) -> None:
        self.txn: List[dict] = []
        self.users: List[dict] = []
        self.merch: List[dict] = []
        self.by_device: List[dict] = []
        self.top: List[dict] = []
        self.geo: Dict[str, dict] = {}
        self.seen = 0
        self.parsed = 0
        self.skipped: Counter = Counter()
        self.skips: List[Tuple[str, str]] = []

    def skip(self, rel: str, reason: str) -> None:
        self.skipped[reason] += 1
        if len(self.skips) < 40:
            self.skips.append((rel, reason))

    def note_geo(self, geo_id: str, level: str, slug: Optional[str],
                 district: Optional[str]) -> None:
        if geo_id not in self.geo:
            self.geo[geo_id] = {"geo_id": geo_id, "level": level,
                                "state_slug": slug, "state_name": label_of(slug) if slug else None,
                                "district_name": district}


def parse_file(col: Collector, path: Path) -> None:
    rel = "/".join(path.relative_to(CACHE).parts)
    col.seen += 1
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        col.skip(rel, f"unreadable json ({type(exc).__name__})")
        return
    data = payload.get("data")
    if data is None:
        col.skip(rel, "data is null")
        return

    parts = path.relative_to(CACHE).parts
    view, metric = parts[0], parts[1]
    try:
        year, quarter = int(parts[-2]), int(path.stem)
    except ValueError:
        col.skip(rel, "path has no year/quarter")
        return
    level = _level(parts, rel)
    slug = _state_slug_from(rel)
    tq = {"year": year, "quarter": quarter}

    if view == "aggregated":
        if metric == "transaction":
            rows = data.get("transactionData") or []
            if not rows:
                col.skip(rel, "aggregated/transaction: empty transactionData")
                return
            gid = geo_state(slug) if slug else geo_country()
            col.note_geo(gid, "state" if slug else "country", slug, None)
            for e in rows:
                for pi in (e.get("paymentInstruments") or []):
                    col.txn.append({"geo_id": gid, **tq,
                                    "category": e.get("name"),
                                    "instrument": pi.get("type"),
                                    "txn_count": pi.get("count"),
                                    "txn_amount_inr": pi.get("amount")})
        else:
            agg = data.get("aggregated") or {}
            if "registeredCount" not in agg:
                col.skip(rel, f"aggregated/{metric}: no registeredCount")
                return
            gid = geo_state(slug) if slug else geo_country()
            col.note_geo(gid, "state" if slug else "country", slug, None)
            rec = {"geo_id": gid, **tq}
            if metric == "user":
                col.users.append({**rec, "registered_users": agg.get("registeredCount"),
                                  "app_opens": agg.get("appOpens")})
            else:
                col.merch.append({**rec, "registered_merchants": agg.get("registeredCount")})
            # usersByDevice appears in no file at this commit; handled if it returns
            for d in (data.get("usersByDevice") or agg.get("usersByDevice") or []):
                col.by_device.append({**rec, "brand": d.get("brand"),
                                      "count": d.get("count"), "pct": d.get("percentage")})

    elif view == "map":
        if metric == "transaction":
            entries = data.get("hoverDataList") or []
            if not entries:
                col.skip(rel, "map/transaction: empty hoverDataList")
                return
            for e in entries:
                name = e.get("name") or ""
                for m in (e.get("metric") or []):
                    if level == "district":
                        dn = norm_district(name)
                        gid = geo_district(slug, dn)
                        col.note_geo(gid, "district", slug, dn)
                    else:
                        s2 = slug_of(name)
                        gid = geo_state(s2)
                        col.note_geo(gid, "state", s2, None)
                    col.txn.append({"geo_id": gid, **tq, "category": "TOTAL",
                                    "instrument": m.get("type"),
                                    "txn_count": m.get("count"),
                                    "txn_amount_inr": m.get("amount")})
        else:
            hover = data.get("hoverData") or {}
            if not hover:
                col.skip(rel, f"map/{metric}: empty hoverData")
                return
            for name, v in hover.items():
                v = v or {}
                if level == "district":
                    dn = norm_district(name)
                    gid = geo_district(slug, dn)
                    col.note_geo(gid, "district", slug, dn)
                else:
                    s2 = slug_of(name)
                    gid = geo_state(s2)
                    col.note_geo(gid, "state", s2, None)
                rec = {"geo_id": gid, **tq}
                if metric == "user":
                    col.users.append({**rec, "registered_users": v.get("registeredCount"),
                                      "app_opens": v.get("appOpens")})
                else:
                    col.merch.append({**rec, "registered_merchants": v.get("registeredCount")})

    elif view == "top":
        got = False
        for key, kind in (("states", "state"), ("districts", "district")):
            for e in (data.get(key) or []):          # null, not missing, on state files
                got = True
                raw = e.get("entityName") or e.get("name")
                m = e.get("metric") or {}
                col.top.append({
                    "scope_geo_id": geo_state(slug) if slug else geo_country(),
                    "metric": metric, "rank_within": kind, **tq,
                    "entity_raw": raw,
                    "entity_norm": norm_district(raw) if kind == "district" else slug_of(raw),
                    "count": m.get("count") if m else e.get("registeredCount"),
                    "amount_inr": m.get("amount") if m else None})
        if not got:
            col.skip(rel, f"top/{metric}: both states and districts empty")
            return
    else:
        col.skip(rel, f"unknown view {view!r}")
        return

    col.parsed += 1


# ---------------------------------------------------------------- build

def build(dry_run: bool = False) -> dict:
    import pandas as pd

    if not CACHE.exists():
        raise SystemExit(f"{CACHE} missing - run `python -m india.pulse_fetch` first")

    col = Collector()
    for p in sorted(CACHE.rglob("*.json")):
        parse_file(col, p)

    print(f"[INFO] files seen {col.seen:,} | parsed {col.parsed:,} | "
          f"skipped {sum(col.skipped.values()):,}")
    for reason, n in col.skipped.most_common():
        print(f"  [SKIP] {n:,} x {reason}")
    for rel, reason in col.skips[:10]:
        print(f"         e.g. {rel} -> {reason}")

    txn = pd.DataFrame(col.txn)
    users = pd.DataFrame(col.users).drop_duplicates(subset=["geo_id", "year", "quarter"])
    merch = pd.DataFrame(col.merch).drop_duplicates(subset=["geo_id", "year", "quarter"])
    top = pd.DataFrame(col.top)
    geo = pd.DataFrame(sorted(col.geo.values(), key=lambda r: r["geo_id"]))

    qs = sorted({(int(y), int(q)) for y, q in zip(txn.year, txn.quarter)})
    dim_time = pd.DataFrame([{"year": y, "quarter": q,
                              "quarter_start": date.fromisoformat(f"{y}-{QUARTER_SPAN[q][0]}"),
                              "quarter_end": date.fromisoformat(f"{y}-{QUARTER_SPAN[q][1]}")}
                             for y, q in qs])
    cats = sorted(c for c in txn.category.dropna().unique())
    dim_cat = pd.DataFrame([{"category": c,
                             "category_norm": CATEGORY_BUCKET.get(c.lower(), c.lower())}
                            for c in cats])

    print(f"\n[INFO] fact_transactions {len(txn):,} | fact_users {len(users):,} | "
          f"fact_merchants {len(merch):,} | top {len(top):,}")
    print(f"[INFO] dim_geo {len(geo):,} "
          f"({(geo.level=='country').sum()} country / {(geo.level=='state').sum()} state / "
          f"{(geo.level=='district').sum()} district) | dim_time {len(dim_time):,} | "
          f"dim_category {len(dim_cat):,} ({', '.join(cats)})")
    if not col.by_device:
        print("[INFO] fact_users_by_device: NOT WRITTEN - usersByDevice absent from all "
              f"{col.seen:,} files at this commit")

    if dry_run:
        print("\n[DRY-RUN] nothing written")
        return {"seen": col.seen, "parsed": col.parsed, "skipped": sum(col.skipped.values())}

    OUT.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    txn.to_parquet(OUT / "fact_transactions.parquet", index=False)
    users.to_parquet(OUT / "fact_users.parquet", index=False)
    merch.to_parquet(OUT / "fact_merchants.parquet", index=False)
    top.to_parquet(OUT / "fact_top.parquet", index=False)
    geo.to_parquet(OUT / "dim_geo.parquet", index=False)
    dim_time.to_parquet(OUT / "dim_time.parquet", index=False)
    dim_cat.to_parquet(OUT / "dim_category.parquet", index=False)
    if col.by_device:
        pd.DataFrame(col.by_device).to_parquet(OUT / "fact_users_by_device.parquet", index=False)

    # tidy CSVs: joined to readable geography, for pandas and the chat agent
    g = geo.set_index("geo_id")
    t = txn.join(g, on="geo_id")
    t[t.level == "country"][["year", "quarter", "category", "txn_count", "txn_amount_inr"]] \
        .sort_values(["year", "quarter", "category"]) \
        .to_csv(CSV / "transactions_country.csv", index=False)
    t[t.level == "state"][["state_name", "year", "quarter", "category",
                           "txn_count", "txn_amount_inr"]] \
        .sort_values(["state_name", "year", "quarter", "category"]) \
        .to_csv(CSV / "transactions_state.csv", index=False)
    d = t[t.level == "district"][["state_name", "district_name", "year", "quarter",
                                  "txn_count", "txn_amount_inr"]].copy()
    d.columns = ["state", "district", "year", "quarter",
                 "transaction_count", "transaction_amount_inr"]
    d.sort_values(["state", "district", "year", "quarter"]) \
        .to_csv(CSV / "transactions_district.csv", index=False)

    u = users.join(g, on="geo_id")
    u[u.level == "district"][["state_name", "district_name", "year", "quarter",
                              "registered_users", "app_opens"]] \
        .sort_values(["state_name", "district_name", "year", "quarter"]) \
        .to_csv(CSV / "users_district.csv", index=False)
    m = merch.join(g, on="geo_id")
    m[m.level == "district"][["state_name", "district_name", "year", "quarter",
                              "registered_merchants"]] \
        .sort_values(["state_name", "district_name", "year", "quarter"]) \
        .to_csv(CSV / "merchants_district.csv", index=False)

    top[top.rank_within == "state"].to_csv(CSV / "top_states.csv", index=False)
    top[top.rank_within == "district"].to_csv(CSV / "top_districts.csv", index=False)

    sha = None
    if META_CACHE.exists():
        sha = json.loads(META_CACHE.read_text(encoding="utf-8")).get("sha")
    meta = {
        "source": "PhonePe Pulse", "source_url": SOURCE_URL, "branch": "main",
        "commit_sha": sha, "licence": LICENCE, "licence_url": LICENCE_URL,
        "attribution": ATTRIBUTION,
        "pulled_at": date.today().isoformat(),
        "files_seen": col.seen, "files_parsed": col.parsed,
        "files_skipped": sum(col.skipped.values()),
        "skip_reasons": dict(col.skipped),
        "period": {"from": f"{qs[0][0]}Q{qs[0][1]}", "to": f"{qs[-1][0]}Q{qs[-1][1]}"},
        "rows": {"fact_transactions": len(txn), "fact_users": len(users),
                 "fact_merchants": len(merch), "fact_top": len(top),
                 "dim_geo": len(geo), "dim_time": len(dim_time),
                 "dim_category": len(dim_cat)},
        "caveats": [
            "PhonePe only - roughly half of UPI, not the whole market.",
            "District rows are TOTAL only; the P2P/Retail/Utility split exists at state and country level.",
            "Rupee amounts come from the map/hover view; aggregated/* carries counts with no amount at this commit.",
            "usersByDevice and appOpens are absent from every file at this commit.",
            "No pincode level - removed in the repo restructure.",
        ],
    }
    (OUT / "source_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[DONE] wrote {OUT} ({sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())/1e6:.1f} MB)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten the Pulse mirror into data/processed/india/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    build(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
