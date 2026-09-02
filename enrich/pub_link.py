#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolve a free-text data mention to a catalogue entry.

The old publication layer stopped at the model's own words ("GBA/BRP",
"StatLine table on employment"), so `n_cbs_table_ids` counted literal ids in the
text but nothing was ever joined against the actual catalogue. With metadata for
4,868 CBS tables and 7,572 Eurostat datasets on disk, a mention can be resolved
to a real code - which is what makes the evidence checkable.

Three strategies, most reliable first:
  exact_id      the mention contains a catalogue code (83765NED, nama_10r_3gdp)
  fuzzy_title   token overlap against catalogue titles, above a threshold
  register      a known CBS microdata register - deliberately NOT linked, since
                registers are confidential and have no StatLine table
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

CBS_CATALOG = Path("data/processed/statline_catalog.parquet")
EST_CATALOG = Path("data/processed/eurostat_catalog.parquet")
ENRICHED = Path("data/processed/enriched_unified_qwen3-32b.jsonl")

CBS_ID_RE = re.compile(r"\b(\d{4,6}[A-Za-z]{2,4})\b")
EST_ID_RE = re.compile(r"\b([a-z]{2,6}_[a-z0-9_]{2,30})\b", re.I)

# Confidential microdata registers: real, citable, but never StatLine tables.
REGISTERS = {
    "GBA", "BRP", "POLIS", "SPOLIS", "SECMBUS", "SECM", "HOOGSTEOPLTAB",
    "INPATAB", "INHATAB", "VSLGWBTAB", "VEHTAB", "KOPPELTAB", "GBAPERSOON",
    "GBAHUISHOUDENS", "BAANKENMERKEN", "NIETBANEN", "ZVWZORGKOSTEN",
}

_WORD = re.compile(r"[a-z0-9]+")
# Filler that appears in how people *refer* to a table, not in its title.
_STOP = {"the", "a", "an", "of", "on", "by", "in", "for", "and", "to", "from",
         "with", "per", "cbs", "statline", "eurostat", "table", "tabel", "dataset",
         "data", "statistics", "statistiek", "figures", "cijfers", "van", "en"}


def _norm_tokens(s: str, drop_stop: bool = False) -> set:
    t = set(_WORD.findall((s or "").lower()))
    return (t - _STOP) if drop_stop else t


@lru_cache(maxsize=1)
def _catalogues() -> Tuple[Dict[str, dict], List[Tuple[set, str, dict]]]:
    """(code -> entry) exact index, and a token-set list for fuzzy title matching."""
    by_code: Dict[str, dict] = {}
    titles: List[Tuple[set, str, dict]] = []
    if CBS_CATALOG.exists():
        c = pd.read_parquet(CBS_CATALOG)
        for r in c.itertuples():
            e = {"code": str(r.table_id), "title": str(getattr(r, "Title", "") or ""),
                 "publisher": "CBS"}
            by_code[e["code"].upper()] = e
            if e["title"]:
                titles.append((_norm_tokens(e["title"]), e["title"], e))
    # English titles from the enrichment corpus: publications are usually written
    # in English while CBS titles are Dutch, so without these a mention like
    # "weekly deaths by gender and age" can never reach "Overledenen; geslacht
    # en leeftijd, per week".
    if ENRICHED.exists():
        import json as _json
        with ENRICHED.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = _json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                e = {"code": str(r.get("code")), "title": r.get("title_native") or "",
                     "publisher": r.get("publisher") or ""}
                by_code.setdefault(e["code"].upper(), e)
                for t in (r.get("title_en"), r.get("title_native")):
                    if t:
                        titles.append((_norm_tokens(t), t, e))
    if EST_CATALOG.exists():
        c = pd.read_parquet(EST_CATALOG)
        for r in c.itertuples():
            e = {"code": str(r.code), "title": str(getattr(r, "title", "") or ""),
                 "publisher": "ESTAT"}
            by_code[e["code"].upper()] = e
            if e["title"]:
                titles.append((_norm_tokens(e["title"]), e["title"], e))
    return by_code, titles


def _register_hit(mention: str) -> Optional[str]:
    up = re.sub(r"[^A-Z]", "", (mention or "").upper())
    for reg in REGISTERS:
        if reg in up:
            return reg
    return None


def link(mention: str, min_score: float = 0.60) -> Dict[str, Any]:
    """Resolve one mention. Always returns a dict; `match` is None when unresolved."""
    by_code, titles = _catalogues()
    out: Dict[str, Any] = {"mention": mention, "match": None, "code": None,
                           "title": None, "publisher": None, "score": None}
    if not mention:
        return out

    for m in CBS_ID_RE.findall(mention) + EST_ID_RE.findall(mention):
        e = by_code.get(m.upper())
        if e:
            out.update(match="exact_id", code=e["code"], title=e["title"],
                       publisher=e["publisher"], score=1.0)
            return out

    reg = _register_hit(mention)
    if reg:
        out.update(match="register", code=reg, publisher="CBS-microdata", score=1.0)
        return out

    toks = _norm_tokens(mention, drop_stop=True)
    if len(toks) < 2:            # single generic word would match hundreds of titles
        return out
    best, best_score = None, 0.0
    for ttoks_raw, title, e in titles:
        ttoks = ttoks_raw - _STOP
        if len(ttoks) < 2:
            continue
        inter = len(toks & ttoks)
        if inter < 2:            # cheap gate before scoring
            continue
        # Plain Jaccard. Containment was tried and rejected: it scores a short
        # title that is a subset of the mention at 1.0, so "GDP by NUTS 3 region"
        # matched "Regional Population for GDP Calculation by NUTS 3 Region"
        # ahead of the actual GDP table. Stopword removal, not a softer metric,
        # is what makes referring phrases match.
        score = inter / len(toks | ttoks)
        if score > best_score:
            best, best_score = (title, e), score
    if best and best_score >= min_score:
        title, e = best
        out.update(match="fuzzy_title", code=e["code"], title=title,
                   publisher=e["publisher"], score=round(best_score, 3))
    elif best:
        # keep a near-miss visible rather than dropping it silently
        title, e = best
        seq = difflib.SequenceMatcher(None, mention.lower(), title.lower()).ratio()
        if seq >= 0.80:
            out.update(match="fuzzy_title", code=e["code"], title=title,
                       publisher=e["publisher"], score=round(seq, 3))
    return out
