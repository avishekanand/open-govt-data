#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Category sampling for grounded prompts.

Showing the FIRST k categories of a long codelist biases everything downstream:
`nama_10r_3gdp` has 1,814 NUTS regions sorted alphabetically, so the first 12 are
all Albanian - and the model duly wrote four example queries about Tiranë,
Durrës, Lezhë and Elbasan. An evenly-spaced sample covers the range instead, so
a benchmark built from these queries is not skewed to the head of each codelist.
"""
from __future__ import annotations

from typing import List, Sequence, TypeVar

T = TypeVar("T")


def spread_sample(values: Sequence[T], k: int) -> List[T]:
    """Evenly-spaced sample across an ordered codelist, always keeping the first.

    Index 0 is kept deliberately: in both catalogues the leading code is usually
    the aggregate ("EU27_2020", "Totaal", "Nederland"), which is the single most
    informative category for describing what the table holds.
    """
    n = len(values)
    if k <= 0:
        return []
    if n <= k:
        return list(values)
    step = (n - 1) / (k - 1)
    seen, out = set(), []
    for i in range(k):
        idx = min(n - 1, round(i * step))
        if idx not in seen:
            seen.add(idx)
            out.append(values[idx])
    return out
