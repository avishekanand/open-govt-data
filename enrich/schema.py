#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified enrichment schema shared by CBS StatLine and Eurostat.

Two record halves, deliberately separated:

* **deterministic** - coverage, dimensions, category counts, observation counts.
  Computed from the pulled data, never asked of the model. The old pipelines let
  the LLM restate these and it got them wrong (e.g. echoing a stale catalogue
  `dataend` when the data ran two years further).
* **generative** - the doc2query fields the model actually adds value on.
  `LLM_JSON_SCHEMA` constrains exactly these, and nothing else.

`finalize_record()` merges the two halves into one record shape that is identical
across sources, so a single index can serve both.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------- generative --

LLM_REQUIRED_KEYS = {
    "title_en", "enriched_description", "example_queries",
    "potential_applications", "key_dimensions", "topics", "join_keys", "confidence",
}

# NOTE: keep this to the JSON Schema subset that BOTH structured-output backends
# support. vLLM picks the backend itself (backend='auto'); xgrammar accepts
# `uniqueItems` but llguidance rejects the whole grammar with
# "Unimplemented keys: [\"uniqueItems\"]". Uniqueness is enforced in
# finalize_record() instead, where it cannot break generation.
LLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title_en": {"type": "string"},
        "enriched_description": {"type": "string"},
        "example_queries": {"type": "array", "items": {"type": "string"},
                            "minItems": 4, "maxItems": 8},
        "potential_applications": {"type": "array", "items": {"type": "string"},
                                   "minItems": 2, "maxItems": 5},
        "key_dimensions": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"},
                   "minItems": 3, "maxItems": 6},
        "join_keys": {"type": "array", "items": {"type": "string"}},
        "confidence": {
            "type": "object",
            "properties": {"desc": {"type": "number"}, "queries": {"type": "number"}},
            "required": ["desc", "queries"],
            "additionalProperties": False,
        },
    },
    "required": sorted(LLM_REQUIRED_KEYS),
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a data librarian enriching official statistics metadata so it is "
    "discoverable by English-speaking analysts. You are given the dataset's real "
    "dimensions and the actual category values present in the data. "
    "Return STRICT JSON only - no prose, no markdown, no reasoning."
)

USER_TEMPLATE = """\
Enrich this {publisher} dataset for an English-language search index.
{language_note}

dataset id: {code}
title: {title_native}
description: {description}
period dimension: "{period_dim}"   (coverage: {coverage})

Dimensions, with the ACTUAL category values present in the data:
{dimensions_block}

Measures / observed value ranges:
{measures_block}

RULES
- Base "example_queries" ONLY on dimensions, categories and periods listed above.
  Never invent a breakdown (an age band, a sector, a region) that is not shown.
- A dimension with a single category is not a breakdown - do not build queries on it.
- Write in English; keep key {native_lang} domain terms in parentheses where useful.
- "join_keys" and "key_dimensions" MUST use the bare dimension id exactly as
  written above (e.g. "geo", "{period_dim}") - not a prose label, not a renamed
  field, and never an id that is not listed.

Return a JSON object with EXACTLY these keys:
{{
  "title_en": "<concise English title>",
  "enriched_description": "<2-4 sentences: what it contains and how it is broken down>",
  "example_queries": ["<4-6 natural-language questions this dataset can answer>"],
  "potential_applications": ["<2-4 concrete use cases>"],
  "key_dimensions": ["<dimensions a user would filter by>"],
  "topics": ["<3-6 short English topic tags>"],
  "join_keys": ["<dimensions usable as join keys>"],
  "confidence": {{"desc": <0-1>, "queries": <0-1>}}
}}
"""


# ------------------------------------------------------------- deterministic --

def make_deterministic(code: str, publisher: str, title_native: str,
                       source_url: str = "", description: str = "",
                       last_update: str = "", coverage: Optional[Dict] = None,
                       dimensions: Optional[List[Dict]] = None,
                       measures: Optional[List[Dict]] = None,
                       n_observations: Optional[int] = None,
                       grounded: bool = False) -> Dict[str, Any]:
    """The half of the record that comes from the data, not the model."""
    return {
        "code": code,
        "publisher": publisher,             # "CBS" | "ESTAT"
        "source_url": source_url,
        "title_native": title_native,
        "description_native": description,
        "last_update": last_update,
        "coverage": coverage or {"start": None, "end": None, "n_periods": None},
        "dimensions": dimensions or [],     # [{id, name, n_categories, sample}]
        "measures": measures or [],         # [{name, unit, min, max}]
        "n_observations": n_observations,
        "grounded": grounded,               # were real category values available?
    }


def finalize_record(llm_obj: Dict[str, Any], deterministic: Dict[str, Any],
                    model: str) -> Dict[str, Any]:
    """Merge model output onto the deterministic half. Deterministic fields win."""
    rec = dict(llm_obj)
    # Deduplicate list fields order-preservingly: the schema cannot express
    # uniqueItems (see note above), and models do repeat entries - one run
    # returned key_dimensions = [geo, unit, unit, unit, unit, unit, unit].
    for key in ("example_queries", "potential_applications", "key_dimensions",
                "topics", "join_keys"):
        val = rec.get(key)
        if isinstance(val, list):
            seen, uniq = set(), []
            for x in val:
                k = x.strip().lower() if isinstance(x, str) else str(x)
                if k not in seen:
                    seen.add(k)
                    uniq.append(x)
            rec[key] = uniq
    rec.update(deterministic)
    rec["enrichment_model"] = model
    return rec


def unified_keys() -> List[str]:
    return sorted(LLM_REQUIRED_KEYS | {
        "code", "publisher", "source_url", "title_native", "description_native",
        "last_update", "coverage", "dimensions", "measures", "n_observations",
        "grounded", "enrichment_model",
    })
