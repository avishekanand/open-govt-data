#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-match scoring for benchmark items.

Result sets are compared, not query strings: one question admits many correct
queries. Two policies exist in the literature - BIRD requires exact content AND
column order; Spider 2.0 only requires the gold's core information to be present.
Our questions are underspecified natural language, so BIRD-strict would fail
answers that are right, and we default to containment on the gold columns.

Canonicalisation before comparison:
  * column names lower-cased and stripped
  * numeric values rounded (default 3 dp) so float noise does not fail a match
  * rows sorted, so ORDER BY differences do not matter unless `ordered=True`
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def canon(df: pd.DataFrame, ndigits: int = 3) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(float).round(ndigits)
        else:
            out[c] = out[c].astype(str).str.strip()
    return out


def score(pred: pd.DataFrame, gold: pd.DataFrame, *, ordered: bool = False,
          ndigits: int = 3, strict_columns: bool = False) -> Dict[str, Any]:
    """Compare a predicted result set against gold. Returns a verdict dict."""
    p, g = canon(pred, ndigits), canon(gold, ndigits)
    missing = [c for c in g.columns if c not in p.columns]
    if missing:
        return {"match": False, "reason": f"missing gold columns: {missing}"}
    if strict_columns and list(p.columns) != list(g.columns):
        return {"match": False, "reason": "column set/order differs (strict mode)"}

    p = p[list(g.columns)]
    if not ordered:
        p = p.sort_values(list(g.columns)).reset_index(drop=True)
        g = g.sort_values(list(g.columns)).reset_index(drop=True)
    else:
        p = p.reset_index(drop=True)
        g = g.reset_index(drop=True)

    if len(p) != len(g):
        return {"match": False, "reason": f"row count {len(p)} != gold {len(g)}"}
    eq = p.equals(g)
    return {"match": bool(eq), "reason": "" if eq else "cell values differ",
            "n_rows": len(g), "n_cols": len(g.columns)}


def run_item(con, item: Dict[str, Any], sql: str) -> Dict[str, Any]:
    """Execute a candidate query for an item and score it against the snapshot."""
    if not item.get("executable", True):
        return {"match": None, "reason": "item is not executable (microdata tier)"}
    gold = pd.DataFrame(item["answer_snapshot"]["rows"],
                        columns=item["answer_snapshot"]["columns"])
    try:
        pred = con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        return {"match": False, "reason": f"query failed: {type(exc).__name__}: {exc}"}
    return score(pred, gold, ordered=item.get("ordered", False))
