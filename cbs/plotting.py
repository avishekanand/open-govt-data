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
import matplotlib.ticker as mticker
import pandas as pd

# ---- house style ----------------------------------------------------------
CBS_BLUE = "#0b3d63"
PALETTE = ["#0b3d63", "#e4711a", "#2a9d8f", "#a4243b", "#5b8c5a",
           "#6c5b7b", "#c9a227", "#3d5a80", "#bc4749", "#457b9d"]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.5,
    "axes.edgecolor": "#888888",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def _style_ax(ax, ylabel: str = "", source: str = ""):
    """Apply the house style: clean spines, y-grid only, thousands formatting."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0, labelsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color="#444444")
    # thousands separators when values are large
    try:
        ymax = max(abs(v) for v in ax.get_yticks()) if len(ax.get_yticks()) else 0
        if ymax >= 1000:
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    except Exception:
        pass
    if source:
        ax.figure.text(0.99, 0.01, source, ha="right", va="bottom",
                       fontsize=7.5, color="#999999")


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
                       transform: str = "level", source: str = ""):
    """Render an agent answer (long df: year, series, value) as a polished line/bar fig.

    bar + multiple series -> horizontal bars comparing series at their latest year.
    bar + single series   -> vertical bars over years.
    line                  -> one line per series over years.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if plot_df is None or len(plot_df) == 0:
        ax.text(0.5, 0.5, "no data", ha="center"); return fig
    n_series = plot_df["series"].nunique()

    if chart_type == "bar" and n_series > 1:
        latest = plot_df.sort_values("year").groupby("series").tail(1)
        latest = latest.dropna(subset=["value"]).sort_values("value").tail(12)
        labels = [str(s)[:38] for s in latest["series"]]
        bars = ax.barh(labels, list(latest["value"]), color=CBS_BLUE, height=0.72)
        ax.bar_label(bars, fmt=lambda v: f"{v:,.0f}", padding=3, fontsize=8.5, color="#444")
        ax.margins(x=0.12)
        _style_ax(ax, source=source)
        ax.grid(axis="y", visible=False); ax.grid(axis="x", visible=True)
        ax.set_xlabel(ylabel, color="#444")
    elif chart_type == "bar":
        g = plot_df.dropna(subset=["value"]).sort_values("year")
        bars = ax.bar(g["year"], g["value"], color=CBS_BLUE, width=0.7)
        ax.bar_label(bars, fmt=lambda v: f"{v:,.0f}", padding=2, fontsize=8, color="#444")
        ax.set_xlabel("Year"); _style_ax(ax, ylabel, source)
    else:
        for i, (name, g) in enumerate(plot_df.groupby("series")):
            g = g.sort_values("year")
            ax.plot(g["year"], g["value"], marker="o", ms=4.5, lw=2.2,
                    color=PALETTE[i % len(PALETTE)], label=str(name)[:34],
                    markeredgecolor="white", markeredgewidth=0.7)
        if n_series > 1:
            ax.legend(loc="best", ncol=1 if n_series <= 6 else 2)
        if transform == "yoy":
            ax.axhline(0, color="#999", lw=0.8, zorder=0)
        ax.set_xlabel("Year"); _style_ax(ax, ylabel, source)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=10))

    ax.set_title(title[:80], loc="left", pad=12)
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
