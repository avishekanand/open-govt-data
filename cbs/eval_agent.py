#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correctness checks for the CBS data agent.

Runs a labelled set of natural-language questions through cbs.agent.answer and
checks that the chosen table is plausibly correct (its title/topics contain at
least one expected keyword) and, where specified, that the chart type / transform
match. Prints a PASS/FAIL table and an accuracy score.

    python -m cbs.eval_agent                         # all cases, default models
    python -m cbs.eval_agent --model qwen2.5:7b      # chat model
    python -m cbs.eval_agent --limit 4               # first few cases
"""
from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List

from cbs import agent

# Each case: a question + keywords that the CORRECT table's title/topics should
# contain (any-of). Optional expected chart_type / transform. `expect_reject`
# marks questions that SHOULD be rejected (no good table exists).
CASES: List[Dict[str, Any]] = [
    {"q": "year over year of dutch residents going on holiday abroad",
     "keywords": ["vakanties", "holiday", "tourist", "toeris"], "chart": "line", "transform": "yoy"},
    {"q": "which foreign destinations do dutch tourists visit most",
     "keywords": ["vakanties", "bestemming", "destination", "holiday"], "chart": "bar"},
    {"q": "average disposable income of households over time",
     "keywords": ["inkomen", "income", "besteedbaar", "huishouden"], "chart": "line"},
    {"q": "renewable electricity production by source over time",
     "keywords": ["hernieuwbare", "elektriciteit", "renewable", "electricity"], "chart": "line"},
    {"q": "greenhouse gas CO2 emissions by economic sector",
     "keywords": ["emissie", "broeikas", "co2", "emission", "greenhouse"]},
    {"q": "consumer price index since 2000",
     "keywords": ["consumentenprijs", "price index", "cpi", "prijz"], "chart": "line"},
    {"q": "population by age group in the netherlands",
     "keywords": ["bevolking", "population", "leeftijd", "age"]},
    {"q": "number of live births by mother's country of origin",
     "keywords": ["geboren", "births", "geboorte", "herkomst", "moeder"]},
    {"q": "unemployment rate trend",
     "keywords": ["werkloos", "unemploy", "arbeid", "beroepsbevolking", "labour", "labor"]},
    {"q": "milk and dairy production by factories",
     "keywords": ["zuivel", "melk", "dairy", "milk"]},
]


def table_text(table_id: str) -> str:
    row = agent.get_table_row(table_id)
    return " ".join(str(row.get(k, "")) for k in
                    ("title_nl", "title_en", "topics", "enriched_description")).lower()


def check(case: Dict[str, Any], a: "agent.Answer") -> Dict[str, Any]:
    if case.get("expect_reject"):
        ok = a.rejected
        return {"pass": ok, "why": "rejected as expected" if ok else "should have been rejected"}
    if a.rejected:
        return {"pass": False, "why": f"rejected (conf {a.confidence})"}
    if not a.table_id:
        return {"pass": False, "why": a.error or "no table"}
    txt = table_text(a.table_id)
    kw_hit = any(k.lower() in txt for k in case["keywords"])
    chart_ok = ("chart" not in case) or (a.chart_type == case["chart"])
    ok = kw_hit and chart_ok
    why = []
    if not kw_hit:
        why.append("table keywords not matched")
    if not chart_ok:
        why.append(f"chart {a.chart_type}!={case['chart']}")
    return {"pass": ok, "why": "; ".join(why) or "ok", "kw_hit": kw_hit, "chart_ok": chart_ok}


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent correctness checks")
    ap.add_argument("--model", default=agent.MODEL, help="chat model (understand/plan)")
    ap.add_argument("--verify-model", default=agent.VERIFY_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES
    print(f"running {len(cases)} cases | model={args.model} verify={args.verify_model}\n")
    rows, passed = [], 0
    t0 = time.time()
    for i, case in enumerate(cases, 1):
        try:
            a = agent.answer(case["q"], model=args.model, verify_model=args.verify_model)
            res = check(case, a)
        except Exception as exc:  # noqa: BLE001
            a = None
            res = {"pass": False, "why": f"crash: {exc}"}
        passed += int(res["pass"])
        tid = getattr(a, "table_id", None) if a else None
        conf = getattr(a, "confidence", None) if a else None
        mark = "PASS" if res["pass"] else "FAIL"
        print(f"[{mark}] {case['q'][:50]:50} -> {tid}  (conf={conf}) {res['why']}")
        rows.append({"q": case["q"], "pass": res["pass"], "table": tid, "why": res["why"]})

    acc = passed / len(cases) if cases else 0
    print(f"\n=== {passed}/{len(cases)} passed ({acc:.0%}) in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
