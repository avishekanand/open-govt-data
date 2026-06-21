#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit-free plotting helpers for CBS table observations.

Shared by the search app (interactive) and the report generator (static PNGs).
Auto-picks the main measure, a time axis (Perioden) and a breakdown dimension,
and renders a sensible default chart.
"""
from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def period_col(df: pd.DataFrame) -> Optional[str]:
    """Identify the time/period label column, if any."""
    if "Perioden_label" in df.columns:
        return "Perioden_label"
    for c in [c for c in df.columns if c.endswith("_label")]:
        vals = df[c].dropna().astype(str)
        if len(vals) and vals.str.match(r"^\d{4}").mean() > 0.5:
            return c
    return None


def main_measure(df: pd.DataFrame) -> Optional[str]:
    """The measure with the most numeric observations (best default to chart)."""
    g = df.dropna(subset=["value"]).groupby("measure").size().sort_values(ascending=False)
    return g.index[0] if len(g) else None


def plot_answer_figure(plot_df, chart_type: str, title: str, ylabel: str,
                       transform: str = "level"):
    """Render an agent answer (long df: year, series, value) as a line or bar fig.

    bar + multiple series -> compare series at their latest year.
    bar + single series   -> bars over years.
    line                  -> one line per series over years.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    if plot_df is None or len(plot_df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center"); return fig
    n_series = plot_df["series"].nunique()
    if chart_type == "bar":
        if n_series > 1:
            latest = plot_df.sort_values("year").groupby("series").tail(1)
            latest = latest.dropna(subset=["value"]).sort_values("value", ascending=False).head(12)
            ax.barh([str(s)[:36] for s in latest["series"]][::-1], list(latest["value"])[::-1])
            ax.set_xlabel(ylabel)
        else:
            g = plot_df.dropna(subset=["value"]).sort_values("year")
            ax.bar(g["year"], g["value"]); ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
    else:
        for name, g in plot_df.groupby("series"):
            g = g.sort_values("year")
            ax.plot(g["year"], g["value"], marker="o", ms=3, label=str(name)[:34])
        if n_series > 1:
            ax.legend(fontsize=8)
        if transform == "yoy":
            ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("Year"); ax.set_ylabel(ylabel)
    ax.set_title(title[:74]); ax.grid(alpha=.3)
    fig.tight_layout()
    return fig


def auto_plot(df: pd.DataFrame, title: str, out_path: str, max_series: int = 6) -> bool:
    """Render a default chart for a tidy table dataframe to out_path. Returns success."""
    df = df.copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].notna().sum() == 0:
        return False
    measure = main_measure(df)
    if not measure:
        return False
    sub = df[df["measure"] == measure].copy()
    unit = sub["unit"].dropna().iloc[0] if sub["unit"].notna().any() else ""
    pcol = period_col(df)
    dim_cols = [c for c in df.columns if c.endswith("_label") and c != pcol]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    if pcol:
        sub["year"] = pd.to_numeric(sub[pcol].astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
        if dim_cols:
            cat = dim_cols[0]
            latest = sub.dropna(subset=["year"])
            top = (latest.sort_values("year").groupby(cat)["value"].last()
                   .sort_values(ascending=False).head(max_series).index.tolist())
            for name in top:
                s = sub[sub[cat] == name].dropna(subset=["year", "value"]).sort_values("year")
                if not s.empty:
                    ax.plot(s["year"], s["value"], marker="o", ms=3, label=str(name)[:32])
            if top:
                ax.legend(fontsize=7)
        else:
            s = sub.dropna(subset=["year", "value"]).sort_values("year")
            ax.plot(s["year"], s["value"], marker="o", ms=3)
        ax.set_xlabel("Period")
    elif dim_cols:
        cat = dim_cols[0]
        s = (sub.dropna(subset=["value"]).groupby(cat)["value"].sum()
             .sort_values(ascending=False).head(12))
        ax.barh([str(i)[:38] for i in s.index][::-1], s.values[::-1])
    else:
        plt.close(fig)
        return False

    ax.set_ylabel(f"{measure} ({unit})"[:55])
    ax.set_title(title[:72], fontsize=10)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True
