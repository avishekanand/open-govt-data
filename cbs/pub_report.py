#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 of the CBS microdata-use publication layer: human-readable findings.

Reads the downloaded documents and writes a Markdown report — one entry per
document: the link, what the file is about, and which CBS data it uses
(detected StatLine table ids, microdata/StatLine references, and the CBS
project number that links the publication to the microdata environment).

The raw downloads + scraped text stay git-ignored; this MD is the kept artifact.

Input : data/processed/pub/documents.parquet  (from cbs.pub_download)
        data/processed/pub/publication_urls.parquet
Output: data/processed/pub/publication_findings.md

    python -m cbs.pub_report
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PUB = Path("data/processed/pub")


def _about(row) -> str:
    title = (row.get("title") or "").strip()
    snippet = " ".join((row.get("text") or "").split())[:240]
    if title and len(title) > 5:
        return f"**{title}** — {snippet}" if snippet else f"**{title}**"
    return snippet or "_(no extractable text)_"


def _cbs_line(row) -> str:
    bits = []
    ids = (row.get("cbs_table_ids") or "")
    if ids:
        bits.append(f"StatLine table ids detected: `{ids}`")
    if row.get("mentions_microdata"):
        bits.append("mentions microdata / StatLine / CBS")
    proj = row.get("project")
    if proj and str(proj) not in ("nan", "None", ""):
        bits.append(f"CBS project **{proj}**")
    return " · ".join(bits) if bits else "_no explicit CBS table reference found in text_"


def _load_extractions(path: Path) -> dict:
    """url -> {cbs_datasets, data_kind, summary} from the stage-4 LLM pass."""
    if not path.exists():
        return {}
    e = pd.read_parquet(path)
    return {r["url"]: r for _, r in e.iterrows()}


def _dataset_frequency(ext: dict) -> list:
    from collections import Counter
    c = Counter()
    for r in ext.values():
        for d in str(r.get("cbs_datasets") or "").split(","):
            d = d.strip()
            if d and d.lower() not in ("none", "unclear", "n/a"):
                c[d] += 1
    return c.most_common(25)


def build(docs: pd.DataFrame, urls: pd.DataFrame, out: Path,
          ext_path: Path = PUB / "pub_extractions.parquet") -> None:
    ext = _load_extractions(ext_path)
    ok = docs[docs["ok"]].copy() if "ok" in docs.columns else docs.copy()
    # rank: docs with CBS table ids first, then microdata mentions, then text length
    ok["_score"] = (ok.get("n_cbs_table_ids", 0).fillna(0) * 100
                    + ok.get("mentions_microdata", False).astype(int) * 10
                    + (ok.get("text_len", 0).fillna(0) > 500).astype(int))
    ok = ok.sort_values("_score", ascending=False)

    # rank docs with an LLM extraction first
    ok["_has_ext"] = ok["url"].map(lambda u: 1 if u in ext else 0)
    ok = ok.sort_values(["_has_ext", "_score"], ascending=False)

    n_total = len(docs)
    n_ok = int(docs["ok"].sum()) if "ok" in docs.columns else len(docs)
    n_ids = int((docs.get("n_cbs_table_ids", 0).fillna(0) > 0).sum())
    n_micro = int(docs.get("mentions_microdata", False).sum())

    lines = [
        "# CBS microdata-use — publication findings",
        "",
        f"_Generated from {n_total} processed URLs._",
        "",
        "| metric | count |",
        "|---|---|",
        f"| documents processed | {n_total} |",
        f"| downloaded OK (text/PDF) | {n_ok} |",
        f"| mention microdata / StatLine / CBS | {n_micro} |",
        f"| explicit StatLine table-id detected | {n_ids} |",
        "",
        "Each entry: **link** · what it is about · which CBS data it uses "
        "(table ids, microdata mention, CBS project number).",
        "",
    ]

    # Aggregate: which CBS datasets are used most (from the LLM extraction).
    freq = _dataset_frequency(ext)
    if freq:
        lines += [f"## Most-used CBS datasets across {len(ext)} analysed publications", ""]
        lines += ["| CBS dataset / register | publications |", "|---|---|"]
        lines += [f"| {name} | {n} |" for name, n in freq]
        lines += [""]
    lines += ["---", ""]

    for _, r in ok.iterrows():
        url = r["url"]
        final = r.get("final_url") or url
        dom = r.get("domain") or ""
        lines.append(f"### [{dom}]({final})")
        lines.append(f"- **Link:** {url}")
        lines.append(f"- **About:** {_about(r)}")
        ex = ext.get(url)
        if ex is not None:
            ds = (ex.get("cbs_datasets") or "").strip()
            lines.append(f"- **CBS datasets used (LLM):** {ds or '_none identified_'}  _[{ex.get('data_kind','')}]_")
            if ex.get("summary"):
                lines.append(f"- **How used:** {ex['summary']}")
        lines.append(f"- **Signals:** {_cbs_line(r)}")
        rtype = r.get("resource_type") or ""
        lines.append(f"- **Type:** {rtype} · status {r.get('status_code')}")
        lines.append("")

    # brief dead/failed appendix
    bad = docs[~docs["ok"]] if "ok" in docs.columns else docs.iloc[0:0]
    if len(bad):
        lines += ["---", "", f"## Unreachable / no text ({len(bad)})", ""]
        for _, r in bad.head(400).iterrows():
            lines.append(f"- {r['url']} — {r.get('error') or r.get('status_code')}")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] findings: {n_ok} documents ({n_ids} with table ids, {n_micro} mention microdata) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Markdown findings report")
    ap.add_argument("--docs", default=str(PUB / "documents.parquet"), type=Path)
    ap.add_argument("--urls", default=str(PUB / "publication_urls.parquet"), type=Path)
    ap.add_argument("--out", default=str(PUB / "publication_findings.md"), type=Path)
    args = ap.parse_args()
    if not args.docs.exists():
        print(f"[ERR] {args.docs} not found — run cbs.pub_download first.")
        return
    docs = pd.read_parquet(args.docs)
    urls = pd.read_parquet(args.urls) if args.urls.exists() else pd.DataFrame()
    build(docs, urls, args.out)


if __name__ == "__main__":
    main()
