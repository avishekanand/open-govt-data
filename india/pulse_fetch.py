#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mirror the PhonePe Pulse JSON tree to a local cache.

The download is driven by the git tree listing, never by constructing paths from
a year x quarter x state product. Those products are mostly wrong: 2026 has only
Q1-Q2, the `merchant` metric starts later than `transaction`, and the
restructure that replaced `insurance` with `merchant` left different quarter
coverage per metric. Enumerating what the repo actually contains means a missing
quarter is a fact about the data rather than a 404 to retry.

Branch `main`, not `master`: master is stale (stops at 2024, still carries the
old `insurance` category).

    python -m india.pulse_fetch                 # mirror to data/raw/india/
    python -m india.pulse_fetch --report        # count what is there, download nothing

Licence: Community Data License Agreement - Permissive 2.0. The licence and the
commit the mirror was taken at are recorded in source_meta.json by the transform
so a published figure can be traced to an exact revision.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = "PhonePe/pulse"
BRANCH = "main"
TREE_API = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
COMMIT_API = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"
CACHE = Path("data/raw/india/pulse")
TREE_CACHE = Path("data/raw/india/_tree_main.json")
META_CACHE = Path("data/raw/india/_commit_main.json")
LICENCE = "CDLA-Permissive-2.0"


def _session():
    import requests
    s = requests.Session()
    s.headers.update({"User-Agent": "open-govt-data/india-pulse (research)",
                      "Accept": "application/vnd.github+json"})
    return s


def commit_sha(session=None, refresh: bool = False) -> Optional[str]:
    """The commit the mirror corresponds to; cached so the transform is offline."""
    if META_CACHE.exists() and not refresh:
        try:
            return json.loads(META_CACHE.read_text(encoding="utf-8")).get("sha")
        except Exception:  # noqa: BLE001
            pass
    s = session or _session()
    try:
        r = s.get(COMMIT_API, timeout=30)
        r.raise_for_status()
        c = r.json()
        rec = {"sha": c.get("sha"),
               "committed_at": (c.get("commit") or {}).get("committer", {}).get("date")}
        META_CACHE.parent.mkdir(parents=True, exist_ok=True)
        META_CACHE.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec["sha"]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] could not read commit sha: {exc}")
        return None


def tree(session=None, refresh: bool = False) -> List[Dict[str, Any]]:
    """Every JSON blob under data/ in the repo, from the recursive git tree.

    Aborts on a truncated tree rather than mirroring a silent subset: GitHub
    truncates very large trees, and a short mirror would read downstream as
    missing quarters instead of a failed listing.
    """
    if TREE_CACHE.exists() and not refresh:
        payload = json.loads(TREE_CACHE.read_text(encoding="utf-8"))
    else:
        s = session or _session()
        r = s.get(TREE_API, timeout=60)
        r.raise_for_status()
        payload = r.json()
        TREE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        TREE_CACHE.write_text(json.dumps(payload), encoding="utf-8")
    if payload.get("truncated"):
        raise RuntimeError("GitHub truncated the tree listing; cannot mirror completely")
    return [b for b in payload["tree"]
            if b.get("type") == "blob"
            and b["path"].startswith("data/")
            and b["path"].endswith(".json")]


def _get_one(session, path: str) -> tuple:
    dest = CACHE / path
    if dest.exists() and dest.stat().st_size > 0:
        return path, "cached", None
    try:
        r = session.get(RAW_BASE + path, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(r.content)
        tmp.replace(dest)                      # atomic: a killed job leaves no half file
        return path, "downloaded", None
    except Exception as exc:  # noqa: BLE001
        return path, "failed", str(exc)


def mirror(workers: int = 16, report_only: bool = False, refresh: bool = False) -> dict:
    session = _session()
    blobs = tree(session, refresh=refresh)
    total_bytes = sum(b.get("size", 0) for b in blobs)
    print(f"[INFO] tree lists {len(blobs):,} JSON files under data/ "
          f"({total_bytes/1e6:.1f} MB) on branch {BRANCH}")
    sha = commit_sha(session, refresh=refresh)
    print(f"[INFO] commit {sha}")

    present = sum(1 for b in blobs if (CACHE / b["path"]).exists())
    print(f"[INFO] already cached: {present:,} of {len(blobs):,}")
    if report_only:
        return {"listed": len(blobs), "cached": present, "commit": sha}

    t0 = time.time()
    counts = {"downloaded": 0, "cached": 0, "failed": 0}
    failures: List[tuple] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_get_one, session, b["path"]): b["path"] for b in blobs}
        done = 0
        for fut in as_completed(futs):
            path, status, err = fut.result()
            counts[status] += 1
            if status == "failed":
                failures.append((path, err))
            done += 1
            if done % 2000 == 0:
                print(f"  {done:,}/{len(blobs):,}", flush=True)

    print(f"[DONE] downloaded {counts['downloaded']:,} | cached {counts['cached']:,} | "
          f"failed {counts['failed']:,} in {(time.time()-t0)/60:.1f}m -> {CACHE}")
    for path, err in failures[:10]:
        print(f"  [FAIL] {path}: {err}")
    if counts["failed"]:
        print(f"[WARN] {counts['failed']} files missing; re-run to retry (resumable)")
    return {"listed": len(blobs), "commit": sha, **counts}


def main() -> None:
    ap = argparse.ArgumentParser(description="Mirror PhonePe Pulse JSON to data/raw/india/")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--report", action="store_true", help="count only, download nothing")
    ap.add_argument("--refresh", action="store_true", help="re-read the git tree listing")
    args = ap.parse_args()
    res = mirror(workers=args.workers, report_only=args.report, refresh=args.refresh)
    if res.get("failed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
