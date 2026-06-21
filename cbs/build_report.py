#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a self-contained HTML gallery of enriched CBS tables.

For each enriched table it shows: title (NL + EN), the gemma4-generated
example questions, a link to the CBS source — and an auto-generated plot of
the actual observations, so you can eyeball "how the results look" at a glance.

Images are base64-embedded, so the single .html file is fully portable.

    python -m cbs.build_report --limit 15 --out data/processed/cbs_report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from cbs.fetch_table_data import fetch_tidy
from cbs.plotting import auto_plot

ENRICHED = Path("data/processed/cbs_enriched_gemma4.jsonl")
CATALOG = Path("data/processed/statline_catalog.parquet")


def load_enriched(path: Path) -> List[dict]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except Exception:
                pass
    return recs


def obs_counts(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    c = pd.read_parquet(path)
    return dict(zip(c["table_id"].astype(str),
                    pd.to_numeric(c["ObservationCount"], errors="coerce").fillna(0).astype(int)))


def img_tag(png: Path) -> str:
    b64 = base64.b64encode(png.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px"/>'


CARD = """
<div class="card">
  <h2>{code} &middot; {title_nl}</h2>
  <div class="en">🇬🇧 {title_en}</div>
  <p class="desc">{desc}</p>
  <div class="grid">
    <div class="qs">
      <b>Example questions this table answers</b>
      <ul>{questions}</ul>
      <div class="topics">{topics}</div>
      <a href="{src}" target="_blank">CBS table ↗</a>
    </div>
    <div class="plot">{plot}</div>
  </div>
</div>
"""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>CBS enriched tables — questions &amp; plots</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}}
 header{{background:#0b3d63;color:#fff;padding:20px 32px}}
 header h1{{margin:0;font-size:20px}} header p{{margin:4px 0 0;opacity:.85;font-size:13px}}
 .card{{background:#fff;margin:18px 32px;padding:18px 22px;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
 .card h2{{margin:0;font-size:16px}} .en{{color:#0b3d63;font-size:13px;margin:2px 0 6px}}
 .desc{{font-size:13px;color:#444;margin:6px 0 12px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
 .qs ul{{margin:6px 0;padding-left:18px}} .qs li{{font-size:13px;margin:3px 0}}
 .topics{{margin:8px 0;font-size:12px;color:#666}} a{{color:#1763a6;text-decoration:none;font-size:13px}}
 @media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>🇳🇱 CBS enriched tables — generated questions &amp; data plots</h1>
<p>{n} tables · each shows gemma4 doc2query questions and an auto-plot of the real observations</p></header>
{cards}
</body></html>"""


def diversify(recs: List[dict], counts: Dict[str, int], skip_over: int,
              per_family: int = 2, per_topic: int = 3) -> List[dict]:
    """Filter to reasonably-sized tables, then greedily pick for spread: cap how
    many tables share a code family (first 3 chars) or a primary topic, so the
    gallery spans subjects instead of repeating one family."""
    from collections import defaultdict
    cand = [r for r in recs if 0 < counts.get(str(r.get("code")), 0) <= skip_over]
    cand.sort(key=lambda r: counts.get(str(r.get("code")), 0))  # smaller → faster/cleaner
    fam_n: Dict[str, int] = defaultdict(int)
    topic_n: Dict[str, int] = defaultdict(int)
    chosen, deferred = [], []
    for r in cand:
        fam = str(r.get("code"))[:3]
        topic = (r.get("topics") or ["other"])[0].strip().lower()
        if fam_n[fam] < per_family and topic_n[topic] < per_topic:
            chosen.append(r); fam_n[fam] += 1; topic_n[topic] += 1
        else:
            deferred.append(r)
    return chosen + deferred  # spread first; fall back to the rest if we need more


def build(enriched: List[dict], out: Path, limit: int, max_obs: int, skip_over: int) -> None:
    counts = obs_counts(CATALOG)
    enriched = diversify(enriched, counts, skip_over)
    tmp = Path(tempfile.mkdtemp())
    cards, made = [], 0
    for r in enriched:
        if made >= limit:
            break
        code = str(r.get("code"))
        n = counts.get(code, 0)
        if n == 0 or n > skip_over:
            continue
        try:
            df = fetch_tidy(code, max_obs=max_obs)
            png = tmp / f"{code}.png"
            if not auto_plot(df, r.get("title_en") or r.get("title_nl") or code, str(png)):
                print(f"  skip {code}: nothing plottable")
                continue
            qs = "".join(f"<li>{html.escape(q)}</li>" for q in (r.get("example_queries") or []))
            cards.append(CARD.format(
                code=code,
                title_nl=html.escape(r.get("title_nl") or ""),
                title_en=html.escape(r.get("title_en") or ""),
                desc=html.escape(r.get("enriched_description") or ""),
                questions=qs,
                topics="🏷 " + html.escape(", ".join(r.get("topics") or [])),
                src=html.escape(r.get("source_url") or ""),
                plot=img_tag(png),
            ))
            made += 1
            print(f"[{made}/{limit}] {code}  ({n:,} obs)  ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {code}: {exc}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.format(n=made, cards="\n".join(cards)), encoding="utf-8")
    print(f"\n[OK] report with {made} tables -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HTML gallery of enriched CBS tables + plots")
    ap.add_argument("--enriched", default=str(ENRICHED), type=Path)
    ap.add_argument("--out", default="data/processed/cbs_report.html", type=Path)
    ap.add_argument("--limit", type=int, default=15, help="Max tables (successful plots) in report")
    ap.add_argument("--max-obs", type=int, default=60000, help="Cap observations fetched per table")
    ap.add_argument("--skip-over", type=int, default=300000, help="Skip tables with more obs than this")
    args = ap.parse_args()
    recs = load_enriched(args.enriched)
    print(f"[INFO] {len(recs)} enriched tables available; building report of up to {args.limit}")
    build(recs, args.out, args.limit, args.max_obs, args.skip_over)


if __name__ == "__main__":
    main()
