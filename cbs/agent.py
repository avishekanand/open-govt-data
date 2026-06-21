#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational agent over CBS StatLine data (Ollama-driven).

Pipeline for a natural-language question:
  1. understand() - LLM extracts search terms + a transform intent (yoy/index/level)
  2. search_index() - FTS5 term search over the metadata index
  3. plan() - LLM picks the best table + measure (y) + breakdown series, from the
     candidates' REAL measures/dimensions (taken from the index)
  4. execute() - deterministically fetch observations, map the plan onto actual
     columns, compute the transform (e.g. YoY %) in pandas
  5. narrate - a short natural-language answer (from the plan)

Single best table per answer (no cross-table joins yet). Uses the local Ollama
model via $OLLAMA_HOST (default gemma4:latest).

CLI:
  python -m cbs.agent "what is the yoy of dutch residents going on holiday abroad"
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from cbs.fetch_table_data import fetch_tidy
from cbs.plotting import period_col

DB_PATH = Path("data/processed/cbs_search.db")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("MODEL", "gemma4:latest")
JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


# --------------------------------------------------------------------- Ollama
def list_models(host: str = OLLAMA_HOST) -> List[str]:
    """Return the model names installed in the Ollama instance."""
    try:
        r = requests.get(host.rstrip("/") + "/api/tags", timeout=5)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def ollama_json(system: str, user: str, model: str = MODEL, temperature: float = 0.1,
                timeout: int = 180) -> Dict[str, Any]:
    """Call Ollama chat with JSON format and return the parsed object."""
    resp = requests.post(
        OLLAMA_HOST.rstrip("/") + "/api/chat",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_ctx": 8192},
            "stream": False,
            "format": "json",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except Exception:
        m = JSON_RE.search(content or "")
        return json.loads(m.group(0)) if m else {}


# ------------------------------------------------------------------ 1. understand
UNDERSTAND_SYS = (
    "You turn a user's data question about Dutch official statistics (CBS) into a "
    "search specification. The data is Dutch; include Dutch keywords too. Return JSON only."
)
UNDERSTAND_USER = """\
Question: {q}

Return JSON:
{{
  "search_terms": "<5-10 keywords (English AND Dutch) to find the right CBS table>",
  "transform": "<one of: yoy, index100, level>  // yoy = year-over-year change, index100 = indexed to first year=100, level = raw values",
  "intent": "<one sentence on what to chart>"
}}
"""


def understand(q: str, model: str = MODEL) -> Dict[str, Any]:
    out = ollama_json(UNDERSTAND_SYS, UNDERSTAND_USER.format(q=q), model=model)
    out.setdefault("search_terms", q)
    out.setdefault("transform", "level")
    if out["transform"] not in ("yoy", "index100", "level"):
        out["transform"] = "level"
    return out


# --------------------------------------------------------------------- 2. search
def search_index(terms: str, k: int = 6, db_path: Path = DB_PATH) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    toks = re.findall(r"[\w]+", terms, flags=re.UNICODE)
    if not toks:
        return []
    match = " OR ".join(f'"{t}"' for t in toks)
    rows = conn.execute(
        "SELECT t.table_id, t.title_nl, t.title_en, t.topics, t.measures_text, "
        "t.dimensions_text, t.enriched_description, t.example_queries_list, "
        "t.obs_count, bm25(tables_fts) AS score "
        "FROM tables_fts JOIN tables t ON t.rowid = tables_fts.rowid "
        "WHERE tables_fts MATCH ? ORDER BY score LIMIT ?",
        (match, k),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_table_row(table_id: str, db_path: Path = DB_PATH) -> Dict[str, Any]:
    """Full indexed metadata for one table (for the verification step)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    r = conn.execute("SELECT * FROM tables WHERE table_id = ?", (table_id,)).fetchone()
    conn.close()
    return dict(r) if r else {}


# ----------------------------------------------------------------------- 3. plan
PLAN_SYS = (
    "You choose how to chart CBS data to answer a question. You are given candidate "
    "tables with their real measures and dimensions. Pick the single best table and "
    "the exact measure and breakdown to plot. Return JSON only — copy names verbatim."
)
PLAN_USER = """\
Question: {q}
Desired transform: {transform}

Candidate tables (id | title | measures | dimensions):
{candidates}

Match the subject AND direction of the question precisely — e.g. residents going
abroad is NOT the same as visitors arriving; pick the table whose title most
directly describes the question's subject. Prefer a measure that is a count/level
of the thing asked about (not a sub-category) unless a breakdown is requested.

Choose ONE table and return JSON:
{{
  "table_id": "<id from the list>",
  "measure": "<exact measure name to put on the Y axis, copied from that table's measures>",
  "series_dim": "<a dimension title to draw one line per category, or empty string for a single line>",
  "series_values": ["<optional specific categories to show; empty = pick the most relevant>"],
  "transform": "{transform}",
  "chart_type": "<line for trends over time; bar for comparing categories or a single time point>",
  "chart_title": "<short English chart title>",
  "answer": "<1-2 sentence answer to the question, noting the table used>"
}}
"""

VERIFY_SYS = (
    "You are a meticulous data analyst double-checking another model's choice of "
    "CBS table and columns to answer a question. Reason step by step using the "
    "table's description and the questions it is meant to answer, then decide if it "
    "is truly the right table — and switch to a better candidate if not. JSON only."
)
VERIFY_USER = """\
Question: {q}
Requested transform: {transform}

PROPOSED ANSWER:
  table {table_id}: {title}
  description: {description}
  questions this table answers: {queries}
  available measures: {measures}
  dimensions: {dimensions}
  proposed measure (Y): {measure}
  proposed breakdown: {series_dim}
  proposed chart type: {chart_type}

OTHER CANDIDATES:
{candidates}

Think step by step:
1. Does the proposed table's SUBJECT actually match the question (right topic, right
   direction/population, right unit)? Watch for near-misses (inbound vs outbound,
   stock vs flow, persons vs households).
2. Is the proposed MEASURE the correct quantity? If not, name a better measure from
   the table's measures.
3. Is any OTHER candidate a clearly better fit? If so, switch to it.
4. Would a LINE (trend over time) or a BAR (compare categories, or one period) chart
   communicate the answer better?

Return JSON:
{{
  "reasoning": "<2-4 sentences of your step-by-step check>",
  "table_ok": true/false,
  "confidence": <0.0-1.0>,
  "table_id": "<keep, or a better candidate id>",
  "measure": "<keep, or a corrected measure name>",
  "series_dim": "<keep or correct; empty for single series>",
  "chart_type": "line | bar",
  "answer": "<final 1-2 sentence answer>"
}}
"""


def verify(q: str, transform: str, pl: Dict[str, Any], cands: List[Dict[str, Any]],
           model: str = MODEL) -> Dict[str, Any]:
    """Chain-of-thought check that the planned table+columns answer the question."""
    tid = pl.get("table_id")
    row = get_table_row(tid) if tid else {}
    user = VERIFY_USER.format(
        q=q, transform=transform, table_id=tid,
        title=row.get("title_en") or row.get("title_nl") or "",
        description=(row.get("enriched_description") or "")[:600],
        queries=(row.get("example_queries_list") or "").replace("\n", " | ")[:400],
        measures=(row.get("measures_text") or "")[:300],
        dimensions=row.get("dimensions_text") or "",
        measure=pl.get("measure", ""), series_dim=pl.get("series_dim", ""),
        chart_type=pl.get("chart_type", "line"),
        candidates=_fmt_candidates([c for c in cands if c["table_id"] != tid][:6]),
    )
    return ollama_json(VERIFY_SYS, user, model=model)


def _fmt_candidates(cands: List[Dict[str, Any]]) -> str:
    lines = []
    for c in cands:
        meas = (c.get("measures_text") or "")[:300]
        dims = c.get("dimensions_text") or ""
        title = c.get("title_en") or c.get("title_nl")
        lines.append(f"- {c['table_id']} | {title} | measures: {meas} | dims: {dims}")
    return "\n".join(lines)


def plan(q: str, transform: str, cands: List[Dict[str, Any]], model: str = MODEL) -> Dict[str, Any]:
    user = PLAN_USER.format(q=q, transform=transform, candidates=_fmt_candidates(cands))
    out = ollama_json(PLAN_SYS, user, model=model)
    return out


# -------------------------------------------------------------------- 4. execute
def _closest(name: str, options: List[str]) -> Optional[str]:
    if not name or not options:
        return None
    name_l = name.strip().lower()
    for o in options:                       # exact / substring first
        if o and (o.lower() == name_l or name_l in o.lower() or o.lower() in name_l):
            return o
    m = difflib.get_close_matches(name, options, n=1, cutoff=0.5)
    return m[0] if m else None


@dataclass
class Answer:
    question: str
    understanding: Dict[str, Any]
    plan: Dict[str, Any]
    table_id: Optional[str] = None
    title: str = ""
    plot_df: Optional[pd.DataFrame] = None   # long: year, series, value
    ylabel: str = ""
    transform: str = "level"
    chart_type: str = "line"
    narrative: str = ""
    reasoning: str = ""
    confidence: Optional[float] = None
    odata_url: str = ""
    source_url: str = ""
    error: Optional[str] = None


def _apply_transform(s: pd.DataFrame, transform: str) -> pd.DataFrame:
    s = s.sort_values("year").copy()
    if transform == "yoy":
        s["value"] = s["value"].pct_change() * 100.0
    elif transform == "index100":
        base = s["value"].dropna().iloc[0] if s["value"].notna().any() else None
        if base:
            s["value"] = s["value"] / base * 100.0
    return s


def execute(q: str, understanding: Dict[str, Any], pl: Dict[str, Any]) -> Answer:
    ans = Answer(q, understanding, pl, transform=pl.get("transform", understanding.get("transform", "level")))
    ans.chart_type = pl.get("chart_type", "line") if pl.get("chart_type") in ("line", "bar") else "line"
    ans.reasoning = pl.get("reasoning", "")
    ans.confidence = pl.get("confidence")
    tid = pl.get("table_id")
    if not tid:
        ans.error = "No table chosen."
        return ans
    ans.table_id = tid
    ans.narrative = pl.get("answer", "")
    ans.source_url = f"https://opendata.cbs.nl/#/CBS/nl/dataset/{tid}"
    ans.odata_url = f"https://datasets.cbs.nl/odata/v1/CBS/{tid}"

    df = fetch_tidy(tid, max_obs=80000)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    ans.title = pl.get("chart_title") or tid

    # map measure
    measures = df["measure"].dropna().unique().tolist()
    measure = _closest(pl.get("measure", ""), measures) or (measures[0] if measures else None)
    if measure is None:
        ans.error = "No numeric measure to plot."
        return ans
    sub = df[df["measure"] == measure].copy()
    unit = sub["unit"].dropna().iloc[0] if sub["unit"].notna().any() else ""

    pcol = period_col(df)
    if not pcol:
        ans.error = "No time dimension to compute a trend."
        return ans
    sub["year"] = pd.to_numeric(sub[pcol].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
    sub = sub.dropna(subset=["year"])

    # map series breakdown
    dim_cols = [c for c in df.columns if c.endswith("_label") and c != pcol]
    series_dim_title = (pl.get("series_dim") or "").strip()
    series_col = None
    if series_dim_title and dim_cols:
        # match against dim base names
        series_col = _closest(series_dim_title, [c[:-6] for c in dim_cols])
        series_col = f"{series_col}_label" if series_col else None

    frames = []
    if series_col:
        wanted = pl.get("series_values") or []
        avail = sub[series_col].dropna().unique().tolist()
        chosen = [v for v in (_closest(w, avail) for w in wanted) if v]
        if not chosen:  # default: top series by latest value
            chosen = (sub.sort_values("year").groupby(series_col)["value"].last()
                      .sort_values(ascending=False).head(5).index.tolist())
        for name in chosen:
            s = sub[sub[series_col] == name].groupby("year", as_index=False)["value"].sum()
            s = _apply_transform(s, ans.transform)
            s["series"] = str(name)
            frames.append(s)
    else:
        s = sub.groupby("year", as_index=False)["value"].sum()
        s = _apply_transform(s, ans.transform)
        s["series"] = measure
        frames.append(s)

    ans.plot_df = pd.concat(frames, ignore_index=True) if frames else None
    ylab = {"yoy": f"{measure} — YoY %", "index100": f"{measure} (index, first yr=100)",
            "level": f"{measure} ({unit})"}[ans.transform]
    ans.ylabel = ylab[:70]
    return ans


def answer(q: str, model: str = MODEL, do_verify: bool = True) -> Answer:
    understanding = understand(q, model=model)
    # Search with both the LLM-reformulated terms and the raw question for recall.
    cands = search_index(understanding["search_terms"] + " " + q, k=8)
    if not cands:
        cands = search_index(q, k=8)
    if not cands:
        return Answer(q, understanding, {}, error="No tables matched the search terms.")
    pl = plan(q, understanding["transform"], cands, model=model)
    # Chain-of-thought double-check of the chosen table/measure (and chart type).
    if do_verify and pl.get("table_id"):
        v = verify(q, understanding["transform"], pl, cands, model=model)
        if v.get("table_id"):
            pl = {**pl, **{k: v[k] for k in
                  ("table_id", "measure", "series_dim", "chart_type", "answer") if v.get(k)}}
        pl["reasoning"] = v.get("reasoning", "")
        pl["confidence"] = v.get("confidence")
    return execute(q, understanding, pl)


def main() -> None:
    q = " ".join(sys.argv[1:]) or "what is the year over year change of dutch residents going on holiday abroad"
    print(f"Q: {q}\n")
    a = answer(q)
    print("understanding:", json.dumps(a.understanding, ensure_ascii=False))
    print("plan:", json.dumps(a.plan, ensure_ascii=False))
    if a.error:
        print("ERROR:", a.error); return
    print(f"\ntable: {a.table_id}  |  {a.title}")
    print("narrative:", a.narrative)
    if a.reasoning:
        print(f"verify (conf={a.confidence}): {a.reasoning}")
    print(f"transform: {a.transform}  chart: {a.chart_type}  ylabel: {a.ylabel}")
    if a.plot_df is not None:
        print("\nplot data (head):")
        print(a.plot_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
