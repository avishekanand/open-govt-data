#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational UI over CBS StatLine data.

Ask a question in natural language; the agent (your Ollama model) understands it,
searches the metadata index, decides what to put on the x/y axes and how to break
it into series, computes transforms like year-over-year, and plots the answer.

Run:
  streamlit run cbs/chat_app.py
Requires: a built index (python -m cbs.build_search_index) and Ollama running
(gemma4:latest by default; override with $OLLAMA_HOST / $MODEL).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the cbs package importable under `streamlit run cbs/chat_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from cbs import agent

st.set_page_config(page_title="CBS Data Chat", page_icon="💬", layout="wide")

EXAMPLES = [
    "year-over-year change of Dutch residents going on holiday abroad",
    "trend of renewable electricity production by source",
    "how has the consumer price index changed since 2000",
    "CO2 emissions by economic sector over time",
]


def plot_answer(a: "agent.Answer"):
    if a.plot_df is None or a.plot_df.empty:
        st.info("No plottable series for this question.")
        return
    from cbs.plotting import plot_answer_figure
    src = f"Source: CBS {a.table_id}" if a.table_id else ""
    fig = plot_answer_figure(a.plot_df, a.chart_type, a.title, a.ylabel, a.transform, source=src)
    st.pyplot(fig)


def render_answer(a: "agent.Answer"):
    if getattr(a, "rejected", False):
        st.warning(a.narrative)
        if a.reasoning:
            st.caption(f"🧠 {a.reasoning}")
        if a.candidates:
            st.markdown("**Closest candidate tables:**")
            for c in a.candidates:
                st.markdown(f"- `{c['table_id']}` — {c['title']}")
        return
    if a.error and a.plot_df is None:
        st.warning(f"{a.error}")
        if a.understanding:
            st.caption("search terms tried: " + a.understanding.get("search_terms", ""))
        return
    st.markdown(a.narrative or "Here is what I found:")
    plot_answer(a)
    chips = []
    if a.table_id:
        chips.append(f"📊 CBS **{a.table_id}**")
    chips.append(f"chart: `{a.chart_type}`")
    chips.append(f"transform: `{a.transform}`")
    if a.confidence is not None:
        chips.append(f"confidence: `{a.confidence:.0%}`")
    st.caption("  ·  ".join(chips))
    if a.reasoning:
        st.caption(f"🧠 _Verified:_ {a.reasoning}")
    if a.source_url:
        st.markdown(f"[CBS table ↗]({a.source_url})  ·  [OData endpoint ↗]({a.odata_url})")
    with st.expander("🔎 how I answered this (understanding · plan · verification · data)"):
        st.json({"understanding": a.understanding, "plan": a.plan})
        if a.plot_df is not None:
            st.dataframe(a.plot_df, use_container_width=True)


def run_query(q: str, model: str):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.spinner(f"[{model}] Understanding → searching → planning → fetching data…"):
        try:
            a = agent.answer(q, model=model)
        except Exception as exc:  # noqa: BLE001
            a = agent.Answer(q, {}, {}, error=f"{type(exc).__name__}: {exc}")
    st.session_state.messages.append({"role": "assistant", "answer": a})


def main():
    if not agent.DB_PATH.exists():
        st.error(f"Search index not found at `{agent.DB_PATH}`. Build it:\n\n"
                 "`python -m cbs.build_search_index`")
        st.stop()

    st.title("💬 CBS Data Chat")
    st.caption(f"Ask in plain language — answered with charts from CBS StatLine data. "
               f"Model: `{agent.MODEL}` via `{agent.OLLAMA_HOST}`.")

    with st.sidebar:
        st.header("Model")
        installed = agent.list_models()
        if installed:
            # Prefer a lighter model by default if gemma4 is busy with enrichment.
            default = next((m for m in installed if "qwen" in m or "llama" in m), installed[0])
            model = st.selectbox("Ollama model", installed, index=installed.index(default))
        else:
            model = agent.MODEL
            st.warning(f"Ollama not reachable at {agent.OLLAMA_HOST}; using `{model}`.")
        st.caption("Switch models if one is busy. Smaller models answer faster.")
        st.divider()

        st.header("Try an example")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        st.divider()
        st.caption("The agent picks one CBS table per answer and charts the most "
                   "relevant measure over time. Cross-table joins are not yet supported.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(m["content"])
            else:
                render_answer(m["answer"])

    pending = st.session_state.pop("pending", None)
    typed = st.chat_input("e.g. year-over-year values of Dutch residents going for tourism")
    q = typed or pending
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            run_query(q, model)
            render_answer(st.session_state.messages[-1]["answer"])


if __name__ == "__main__":
    main()
