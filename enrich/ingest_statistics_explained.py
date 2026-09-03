#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harvest Eurostat 'Statistics Explained' articles as a seed corpus.

The CBS question corpus comes from publications that happen to cite CBS tables,
and the weak link is resolving those citations - lexical matching recovered only
49 distinct tables. Statistics Explained is structurally better for Eurostat:
every article is written *around* specific datasets and cites their codes
verbatim, so provenance needs no fuzzy matching at all.

    python -m enrich.ingest_statistics_explained --workers 8

Output: data/processed/estat/se_articles.jsonl
        {title, url, text, dataset_codes[], codes_in_catalogue[]}

Codes are validated against data/processed/eurostat_catalog.parquet, so a token
that merely looks like a code but names no real dataset is dropped.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
import requests

API = "https://ec.europa.eu/eurostat/statistics-explained/api.php"
PAGE = "https://ec.europa.eu/eurostat/statistics-explained/index.php?title={t}"
CATALOG = Path("data/processed/eurostat_catalog.parquet")
OUT = Path("data/processed/estat/se_articles.jsonl")
UA = {"User-Agent": "ogd-research/1.0 (metadata benchmark; contact via repo)"}

# Dataset codes look like nama_10r_3gdp / hsw_n2_02. Databrowser links may append
# a "__custom_123" view suffix, which is a saved view of the same dataset.
CODE_RE = re.compile(r"\b([a-z]{2,6}_[a-z0-9_]{2,30})\b")
CUSTOM_RE = re.compile(r"__custom_\d+$")
SKIP_PREFIX = ("Glossary", "Category:", "Template:", "File:", "Help:", "Statistics Explained:")
_lock = threading.Lock()


def list_articles(session: requests.Session, limit: int | None = None) -> List[str]:
    titles, cont = [], {}
    while True:
        p = {"action": "query", "list": "allpages", "apnamespace": "0",
             "aplimit": "500", "format": "json", **cont}
        r = session.get(API, params=p, headers=UA, timeout=60)
        r.raise_for_status()
        d = r.json()
        for a in d.get("query", {}).get("allpages", []):
            t = a["title"]
            if not t.startswith(SKIP_PREFIX) and "Glossary" not in t:
                titles.append(t)
        if limit and len(titles) >= limit:
            return titles[:limit]
        if "continue" not in d:
            return titles
        cont = d["continue"]


def fetch_article(session: requests.Session, title: str, valid: Set[str]) -> Dict[str, Any]:
    p = {"action": "parse", "page": title, "prop": "wikitext", "format": "json"}
    r = session.get(API, params=p, headers=UA, timeout=60)
    r.raise_for_status()
    wt = r.json()["parse"]["wikitext"]["*"]
    raw = {CUSTOM_RE.sub("", c) for c in CODE_RE.findall(wt)}
    # _esms pages are metadata about a dataset, not the dataset itself
    raw = {c[:-5] if c.endswith("_esms") else c for c in raw}
    in_cat = sorted(c for c in raw if c in valid)
    return {"title": title,
            "url": PAGE.format(t=title.replace(" ", "_")),
            "text": wt,
            "text_len": len(wt),
            "dataset_codes": sorted(raw),
            "codes_in_catalogue": in_cat,
            "n_codes": len(in_cat)}


def done_titles(path: Path) -> Set[str]:
    out = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["title"])
                except Exception:  # noqa: BLE001
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest Statistics Explained articles")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=OUT, type=Path)
    args = ap.parse_args()

    valid = set(pd.read_parquet(CATALOG)["code"].astype(str).str.lower()) \
        if CATALOG.exists() else set()
    print(f"[INFO] catalogue codes for validation: {len(valid):,}")

    session = requests.Session()
    titles = list_articles(session, args.limit)
    done = done_titles(args.out)
    todo = [t for t in titles if t not in done]
    print(f"[INFO] {len(titles):,} articles listed, {len(todo):,} to fetch "
          f"({len(done):,} already done)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    t0 = time.time()
    with args.out.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_article, session, t, valid): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
                with _lock:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            except Exception:  # noqa: BLE001
                fail += 1
            if i % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] ok={ok} fail={fail} | {el/60:.1f}m", flush=True)
    print(f"[DONE] ok={ok} fail={fail} in {(time.time()-t0)/60:.1f}m -> {args.out}")


if __name__ == "__main__":
    main()
