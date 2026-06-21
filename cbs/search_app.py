#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit search UI over the CBS StatLine metadata FTS5 index.

A term-matching search engine over all metadata text fields (Dutch title +
description, gemma4 English enrichment, dimensions, measures). Build the index
first with `python -m cbs.build_search_index`.

Run:
  streamlit run cbs/search_app.py
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import streamlit as st

DB_PATH = Path(os.environ.get("CBS_SEARCH_DB", "data/processed/cbs_search.db"))

# Per-column BM25 weights (order matches the FTS5 column order).
FTS_COLS = [
    "title_nl", "title_en", "summary", "enriched_description",
    "example_queries", "topics", "dimensions_text", "measures_text",
]
WEIGHTS = [10.0, 10.0, 3.0, 5.0, 4.0, 6.0, 2.0, 2.0]

st.set_page_config(page_title="CBS Metadata Search", page_icon="🔎", layout="wide")


@st.cache_resource
def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def make_match_query(raw: str, mode: str = "AND") -> str:
    """Turn free text into a safe FTS5 MATCH expression of quoted tokens."""
    tokens = re.findall(r"[\w]+", raw, flags=re.UNICODE)
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens]
    return f" {mode} ".join(quoted)


def search(conn: sqlite3.Connection, raw: str, status, only_enriched, min_obs, limit=50):
    bm25 = f"bm25(tables_fts, {', '.join(str(w) for w in WEIGHTS)})"
    filters, params = [], []
    if status and status != "(all)":
        filters.append("t.status = ?")
        params.append(status)
    if only_enriched:
        filters.append("t.has_enrichment = 1")
    if min_obs:
        filters.append("t.obs_count >= ?")
        params.append(int(min_obs))

    def run(match_expr):
        where = ["tables_fts MATCH ?"] + filters
        sql = (
            f"SELECT t.*, {bm25} AS score, "
            f"snippet(tables_fts, -1, '<mark>', '</mark>', ' … ', 12) AS snip "
            f"FROM tables_fts JOIN tables t ON t.rowid = tables_fts.rowid "
            f"WHERE {' AND '.join(where)} ORDER BY score LIMIT ?"
        )
        return conn.execute(sql, [match_expr, *params, limit]).fetchall()

    # Try AND first (precision), fall back to OR (recall).
    rows = run(make_match_query(raw, "AND"))
    if not rows:
        rows = run(make_match_query(raw, "OR"))
    return rows


def browse(conn, status, only_enriched, min_obs, limit=50):
    filters, params = [], []
    if status and status != "(all)":
        filters.append("status = ?"); params.append(status)
    if only_enriched:
        filters.append("has_enrichment = 1")
    if min_obs:
        filters.append("obs_count >= ?"); params.append(int(min_obs))
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    return conn.execute(
        f"SELECT * FROM tables {where} ORDER BY obs_count DESC LIMIT ?", [*params, limit]
    ).fetchall()


def stats(conn):
    total = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    enr = conn.execute("SELECT COUNT(*) FROM tables WHERE has_enrichment=1").fetchone()[0]
    statuses = [r[0] for r in conn.execute(
        "SELECT DISTINCT status FROM tables WHERE status<>'' ORDER BY status").fetchall()]
    return total, enr, statuses


def render_card(r):
    title_en = r["title_en"]
    head = f"**{r['table_id']} · {r['title_nl']}**"
    st.markdown(head)
    if title_en:
        st.caption(f"🇬🇧 {title_en}")
    if r["snip"]:
        st.markdown(r["snip"], unsafe_allow_html=True)
    elif r["enriched_description"]:
        st.write(r["enriched_description"])
    elif r["summary"]:
        st.write(r["summary"][:280] + ("…" if len(r["summary"]) > 280 else ""))

    bits = []
    if r["topics"]:
        bits.append("🏷 " + r["topics"])
    bits.append(f"📐 {r['n_dimensions']} dims · {r['n_measures']} measures")
    if r["obs_count"]:
        bits.append(f"📊 {r['obs_count']:,} obs")
    if r["status"]:
        bits.append(f"· {r['status']}")
    try:
        bits.append(f"· score {r['score']:.1f}")
    except (IndexError, KeyError):
        pass
    st.caption("  ".join(bits))
    st.markdown(f"[CBS table]({r['source_url']}) · [OData endpoint]({r['odata_url']})")
    st.divider()


def main():
    if not DB_PATH.exists():
        st.error(f"Index not found at `{DB_PATH}`. Build it first:\n\n"
                 "`python -m cbs.build_search_index`")
        st.stop()
    conn = get_conn(str(DB_PATH))
    total, enr, statuses = stats(conn)

    st.title("🔎 CBS Metadata Search")
    st.caption("Term-matching search over StatLine table metadata — Dutch titles & "
               "descriptions, gemma4 English enrichment, dimensions and measures.")

    with st.sidebar:
        st.header("Filters")
        status = st.selectbox("Status", ["(all)"] + statuses, index=0)
        only_enriched = st.checkbox("Only AI-enriched tables", value=False)
        min_obs = st.number_input("Min. observation count", min_value=0, value=0, step=1000)
        st.divider()
        st.metric("Tables indexed", f"{total:,}")
        st.metric("With enrichment", f"{enr:,}")
        st.caption(f"DB: `{DB_PATH}`")

    q = st.text_input("Search", placeholder="e.g. income neighbourhood · vakanties · werkgelegenheid lonen")

    if q.strip():
        rows = search(conn, q, status, only_enriched, min_obs)
        st.subheader(f"{len(rows)} results for “{q}”")
        if not rows:
            st.info("No matches. Try fewer or different terms.")
        for r in rows:
            render_card(r)
    else:
        st.subheader("Browse — largest tables")
        for r in browse(conn, status, only_enriched, min_obs):
            render_card(r)


if __name__ == "__main__":
    main()
