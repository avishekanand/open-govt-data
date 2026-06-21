#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 of the CBS microdata-use publication layer.

Downloads each unique public URL from the workbook, follows redirects, extracts
text (HTML via BeautifulSoup, PDF via pypdf), and captures signals of how the
publication used CBS data (table-id mentions, microdata/StatLine references,
the CBS project number). Polite + concurrent + resumable.

Input : data/processed/pub/publication_urls.parquet  (from cbs.pub_ingest)
Output: data/processed/pub/documents.parquet          (one row per URL, incremental)
        data/raw/pub_docs/<hash>.pdf                   (downloaded PDFs)

    python -m cbs.pub_download                 # all URLs (resumable)
    python -m cbs.pub_download --limit 50      # quick test
"""
from __future__ import annotations

import argparse
import hashlib
import io
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

warnings.filterwarnings("ignore")  # quiet pypdf / urllib noise

PUB = Path("data/processed/pub")
URLS = PUB / "publication_urls.parquet"
DOCS = PUB / "documents.parquet"
RAW = Path("data/raw/pub_docs")

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/pdf,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en,nl;q=0.8",
}
MAX_TEXT = 40000
CBS_TABLE_RE = re.compile(r"\b\d{4,5}[A-Z]{2,3}\b")          # e.g. 83765NED
MICRODATA_RE = re.compile(r"microdata|micro-data|remote access|cbs\b|statline", re.I)


def _hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def extract_html(content: bytes) -> Dict[str, str]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    text = re.sub(r"\s+\n", "\n", soup.get_text("\n", strip=True))
    return {"title": title, "text": text[:MAX_TEXT]}


def extract_pdf(content: bytes, url: str) -> Dict[str, str]:
    from pypdf import PdfReader
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{_hash(url)}.pdf"
    path.write_bytes(content)
    reader = PdfReader(io.BytesIO(content))
    pages = reader.pages[:40]
    text = "\n".join((p.extract_text() or "") for p in pages)
    title = ""
    try:
        title = (reader.metadata or {}).get("/Title", "") or ""
    except Exception:
        pass
    return {"title": str(title), "text": text[:MAX_TEXT], "path": str(path)}


def signals(text: str, project: str) -> Dict[str, Any]:
    ids = sorted(set(CBS_TABLE_RE.findall(text or "")))
    return {
        "cbs_table_ids": ids,
        "n_cbs_table_ids": len(ids),
        "mentions_microdata": bool(MICRODATA_RE.search(text or "")),
        "mentions_project": bool(project and str(project) in (text or "")),
    }


def _first_project(row: Dict[str, Any]):
    projs = row.get("projects")
    try:
        return projs[0] if projs is not None and len(projs) else None
    except Exception:
        return None


def fetch_one(row: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = row["url"]
    out: Dict[str, Any] = {
        "url": url, "resource_type": row.get("resource_type"), "domain": row.get("domain"),
        "project": _first_project(row),
        "final_url": None, "status_code": None, "content_type": None, "ok": False,
        "title": "", "text": "", "text_len": 0, "path": None, "error": None,
        "cbs_table_ids": [], "n_cbs_table_ids": 0,
        "mentions_microdata": False, "mentions_project": False,
    }
    if not re.match(r"^https?://[^/]+\.", url):     # skip junk like 'geen' / bare strings
        out["error"] = "invalid url"
        return out
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True, stream=True)
        out["status_code"] = r.status_code
        out["final_url"] = r.url
        ctype = r.headers.get("Content-Type", "").lower()
        out["content_type"] = ctype
        if r.status_code >= 400:
            out["error"] = f"HTTP {r.status_code}"
            return out
        content = r.content  # realize body (cap implicitly by server)
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            ex = extract_pdf(content, url)
            out["path"] = ex["path"]
        elif "html" in ctype or "xml" in ctype or not ctype:
            ex = extract_html(content)
        else:
            ex = {"title": "", "text": ""}
        out["title"] = ex.get("title", "")
        out["text"] = ex.get("text", "")
        out["text_len"] = len(out["text"])
        out["ok"] = out["text_len"] > 0 or out["path"] is not None
        out.update(signals(out["text"], out["project"]))
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
    return out


def load_done(path: Path) -> set:
    if path.exists():
        try:
            return set(pd.read_parquet(path, columns=["url"])["url"])
        except Exception:
            return set()
    return set()


def append(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    for c in ("cbs_table_ids",):
        df[c] = df[c].apply(lambda v: ",".join(v) if isinstance(v, list) else v)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + extract publication documents")
    ap.add_argument("--urls", default=str(URLS), type=Path)
    ap.add_argument("--out", default=str(DOCS), type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=25)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=50, help="Flush to parquet every N")
    args = ap.parse_args()

    todo = pd.read_parquet(args.urls).to_dict("records")
    done = load_done(args.out)
    todo = [r for r in todo if r["url"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[INFO] {len(todo)} URLs to fetch ({len(done)} already done), {args.workers} workers")

    buf: List[Dict[str, Any]] = []
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, r, args.timeout): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            buf.append(res)
            ok += int(res["ok"]); fail += int(not res["ok"])
            if i % args.batch == 0:
                append(buf, args.out); buf = []
                print(f"[{i}/{len(todo)}] ok={ok} fail={fail}  (flushed)")
    append(buf, args.out)
    print(f"\n[DONE] fetched {len(todo)}: ok={ok} fail={fail} -> {args.out}")


if __name__ == "__main__":
    main()
