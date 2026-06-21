#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 of the CBS microdata-use publication layer.

Parses the publications workbook (public evidence of how CBS microdata has been
used), extracts + normalizes + classifies + dedups the URLs. No network calls.

Input : data/raw/Publications_overview_internet_May_26.xlsx
Output:
  data/processed/pub/publication_records.parquet   (one row per publication)
  data/processed/pub/publication_urls.parquet      (one row per unique URL)

    python -m cbs.pub_ingest
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

WORKBOOK = Path("data/raw/Publications_overview_internet_May_26.xlsx")
OUT = Path("data/processed/pub")

URL_COLS = {"URL publication": "publication_url", "Published code or scripts": "code_url"}


def classify(url: str) -> str:
    u = url.lower()
    host = urlparse(u).netloc
    if "github.com" in host:
        return "github"
    if "zenodo.org" in host:
        return "zenodo"
    if "doi.org" in host or u.startswith("10."):
        return "doi"
    if u.endswith(".pdf") or "/pdf" in u:
        return "pdf"
    if u.endswith((".xlsx", ".xls", ".csv")):
        return "data"
    if u.endswith(".zip"):
        return "zip"
    return "html"


def normalize(url: str) -> str:
    url = str(url).strip()
    if url and not re.match(r"^https?://", url, re.I):
        if url.startswith("10."):       # bare DOI
            url = "https://doi.org/" + url
        elif "." in url:
            url = "https://" + url
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse CBS publications workbook")
    ap.add_argument("--workbook", default=str(WORKBOOK), type=Path)
    ap.add_argument("--out", default=str(OUT), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.workbook)
    df = df.reset_index().rename(columns={"index": "record_id"})

    # Records table (one row per publication).
    rec = df.rename(columns={
        "Institute": "institute", "Project": "project", "Publication title": "title",
        "Author(s)": "authors", "Publication date": "pub_date", "Publisher": "publisher",
        "URL publication": "publication_url", "Published code or scripts": "code_url",
    })
    rec_cols = ["record_id", "institute", "project", "title", "authors",
                "pub_date", "publisher", "publication_url", "code_url"]
    rec = rec[[c for c in rec_cols if c in rec.columns]]
    rec.to_parquet(args.out / "publication_records.parquet", index=False)

    # URLs table (one row per unique URL, with the records referencing it).
    rows = []
    for _, r in df.iterrows():
        for col, kind in URL_COLS.items():
            raw = r.get(col)
            if pd.isna(raw) or not str(raw).strip():
                continue
            url = normalize(raw)
            rows.append({
                "url": url,
                "url_kind": kind,                       # publication_url | code_url
                "resource_type": classify(url),
                "domain": urlparse(url).netloc.lower().replace("www.", ""),
                "record_id": r["record_id"],
                "institute": r.get("Institute"),
                "project": r.get("Project"),
                "title": r.get("Publication title"),
            })
    urls = pd.DataFrame(rows)
    # dedup URLs, keep the list of referencing records
    agg = (urls.groupby("url")
           .agg(url_kind=("url_kind", "first"),
                resource_type=("resource_type", "first"),
                domain=("domain", "first"),
                n_records=("record_id", "nunique"),
                record_ids=("record_id", lambda s: sorted(set(s))),
                projects=("project", lambda s: sorted({str(x) for x in s if pd.notna(x)})))
           .reset_index())
    agg.to_parquet(args.out / "publication_urls.parquet", index=False)

    print(f"[OK] records: {len(rec)} -> {args.out/'publication_records.parquet'}")
    print(f"[OK] unique URLs: {len(agg)} (of {len(urls)} mentions) -> {args.out/'publication_urls.parquet'}")
    print("[INFO] by resource_type:", agg["resource_type"].value_counts().to_dict())
    print("[INFO] top domains:", agg["domain"].value_counts().head(8).to_dict())


if __name__ == "__main__":
    main()
