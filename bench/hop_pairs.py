#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build grounded two-hop dataset pairs, in tiers of evidence strength.

A two-hop question is only worth having if the join is one someone actually
makes. Three tiers of evidence, strongest first:

  tier 1  PROXIMITY   both datasets cited within the same passage of an article.
                      The article is relating them right there.
  tier 2  CO-CITATION both cited in the same article, but far apart. The article
                      used both; whether it related them is unproven.
  tier 3  ATTESTED CROSSING
                      never co-cited, but the DOMAIN crossing they represent
                      (e.g. health x income) is attested elsewhere in the corpus.
                      Co-occurrence teaches which kinds of tables get combined,
                      so a pair can inherit that evidence without being co-cited.

All tiers additionally require >= 2 shared joinable dimensions and DIFFERENT
Eurostat domains - joining "fatal accidents" to "non-fatal accidents" is
comparing two columns of one statistic, not a hop.

    python -m bench.hop_pairs --report
    python -m bench.hop_pairs            # -> data/processed/benchmark/hop_pairs.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILES = Path("data/processed/field_profiles_estat.jsonl")
ARTICLES = Path("data/processed/estat/se_articles.jsonl")
CONSTRUCTED = Path("data/processed/benchmark/constructed_se.jsonl")
OUT = Path("data/processed/benchmark/hop_pairs.jsonl")

JOIN_KEYS = {"geo", "time", "nace_r2", "sex", "age", "isced11", "citizen", "c_birth"}
PROXIMITY = 1500          # characters; roughly a section
DOMAIN_NAME = {
    "nama": "national accounts", "naida": "national accounts (international)",
    "lfsa": "labour force", "lfst": "labour force (regional)", "lfsi": "labour force indicators",
    "demo": "demography", "educ": "education", "edat": "educational attainment",
    "migr": "migration", "hlth": "health", "hsw": "health & safety at work",
    "ilc": "income & living conditions", "nrg": "energy", "env": "environment",
    "gov": "government finance", "sbs": "business structure", "isoc": "digital economy",
    "tour": "tourism", "tran": "transport", "prc": "prices", "ext": "external trade",
    "sdg": "sustainable development", "earn": "earnings", "tps": "key indicators",
}


def domain(code: str) -> str:
    return re.split(r"[_\d]", code.lower())[0]


def load_profiles() -> Dict[str, dict]:
    return {json.loads(l)["code"].lower(): json.loads(l) for l in PROFILES.open(encoding="utf-8")}


def shared_keys(a: dict, b: dict) -> List[str]:
    return sorted((set(a.get("fields") or {}) & set(b.get("fields") or {})) & JOIN_KEYS)


def min_distance(text: str, x: str, y: str) -> Optional[int]:
    xs = [m.start() for m in re.finditer(re.escape(x), text, re.I)]
    ys = [m.start() for m in re.finditer(re.escape(y), text, re.I)]
    if not xs or not ys:
        return None
    return min(abs(i - j) for i in xs for j in ys)


URLISH = re.compile(r"https?://|databrowser/bookmark|\{\{|\[\[Image:|^:+", re.M)
SENTENCE = re.compile(r"[a-z]{3,}[^.!?]{20,}[.!?]\s+[A-Z]")


def is_prose(pas: str) -> bool:
    """Reject source lists and navigation trees.

    Proximity alone catches the reference block at the foot of every article -
    two dataset codes 300 characters apart inside a nested list of databrowser
    links are not being *related*, merely listed. Real analysis is prose.
    """
    if len(pas) < 200:
        return False
    urlish = len(URLISH.findall(pas))
    if urlish >= 3:
        return False
    if len(SENTENCE.findall(pas)) < 2:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in pas)
    return letters / max(len(pas), 1) > 0.75


def passage(text: str, x: str, y: str, width: int = 900) -> str:
    d = None
    best = None
    for i in [m.start() for m in re.finditer(re.escape(x), text, re.I)]:
        for j in [m.start() for m in re.finditer(re.escape(y), text, re.I)]:
            if d is None or abs(i - j) < d:
                d, best = abs(i - j), (min(i, j), max(i, j))
    if not best:
        return ""
    a = max(0, best[0] - 200)
    return re.sub(r"\s+", " ", text[a: best[1] + width]).strip()[: width * 2]


def build(report_only: bool = False) -> None:
    prof = load_profiles()
    arts = [json.loads(l) for l in ARTICLES.open(encoding="utf-8")]

    pairs: Dict[tuple, dict] = {}
    crossing_counts: Counter = Counter()

    for a in arts:
        codes = [c for c in (a.get("codes_in_catalogue") or [])[:12] if c.lower() in prof]
        for x, y in itertools.combinations(sorted(set(codes)), 2):
            px, py = prof[x.lower()], prof[y.lower()]
            if domain(x) == domain(y):
                continue
            keys = shared_keys(px, py)
            if len(keys) < 2:
                continue
            dist = min_distance(a["text"], x, y)
            pas = passage(a["text"], x, y) if (dist is not None and dist <= PROXIMITY) else ""
            # proximity is necessary but not sufficient: the passage must be prose
            tier = 1 if (pas and is_prose(pas)) else 2
            crossing_counts[tuple(sorted((domain(x), domain(y))))] += 1
            k = (x.lower(), y.lower())
            prev = pairs.get(k)
            if prev is None or tier < prev["tier"]:
                pairs[k] = {
                    "a": x, "b": y, "tier": tier, "distance": dist, "join_keys": keys,
                    "domain_a": DOMAIN_NAME.get(domain(x), domain(x)),
                    "domain_b": DOMAIN_NAME.get(domain(y), domain(y)),
                    "article": a["title"], "article_url": a["url"],
                    "passage": pas if tier == 1 else "",
                    "title_a": px.get("title_en"), "title_b": py.get("title_en"),
                    "evidence": ("cited in the same passage" if tier == 1
                                 else "cited in the same article"),
                }

    # tier 3: pairs never co-cited, but whose DOMAIN crossing is attested elsewhere
    pool = set()
    if CONSTRUCTED.exists():
        with CONSTRUCTED.open(encoding="utf-8") as fh:
            for line in fh:
                pool.add(json.loads(line)["gold_dataset"]["code"].lower())
    attested = {c for c, n in crossing_counts.items() if n >= 5}
    t3 = 0
    pool_l = sorted(pool)
    for x, y in itertools.combinations(pool_l, 2):
        if (x, y) in pairs:
            continue
        if domain(x) == domain(y):
            continue
        if tuple(sorted((domain(x), domain(y)))) not in attested:
            continue
        px, py = prof.get(x), prof.get(y)
        if not px or not py:
            continue
        keys = shared_keys(px, py)
        if len(keys) < 2:
            continue
        t3 += 1
        if t3 <= 20000:
            pairs[(x, y)] = {
                "a": x, "b": y, "tier": 3, "distance": None, "join_keys": keys,
                "domain_a": DOMAIN_NAME.get(domain(x), domain(x)),
                "domain_b": DOMAIN_NAME.get(domain(y), domain(y)),
                "article": None, "article_url": None, "passage": "",
                "title_a": px.get("title_en"), "title_b": py.get("title_en"),
                "evidence": "domain crossing attested elsewhere in the corpus",
            }

    by_tier = Counter(p["tier"] for p in pairs.values())
    print(f"tier 1  same passage      : {by_tier[1]:,}")
    print(f"tier 2  same article      : {by_tier[2]:,}")
    print(f"tier 3  attested crossing : {by_tier[3]:,}")
    print(f"\nmost attested domain crossings (from co-citation):")
    for (a, b), n in crossing_counts.most_common(8):
        print(f"  {n:4d}  {DOMAIN_NAME.get(a,a):26s} x {DOMAIN_NAME.get(b,b)}")
    if report_only:
        print("\n--- tier 1 examples (the article relates them in one passage) ---")
        for p in [q for q in pairs.values() if q["tier"] == 1][:5]:
            print(f"\n  {p['article'][:60]}  (gap {p['distance']} chars)")
            print(f"    A {p['a']}: {str(p['title_a'])[:52]}   [{p['domain_a']}]")
            print(f"    B {p['b']}: {str(p['title_b'])[:52]}   [{p['domain_b']}]")
            print(f"    join on {p['join_keys']}")
            print(f"    passage: {p['passage'][:180]}…")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for p in sorted(pairs.values(), key=lambda q: (q["tier"], q["a"])):
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n[DONE] {len(pairs):,} pairs -> {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Tiered two-hop dataset pairs")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    build(report_only=args.report)


if __name__ == "__main__":
    main()
