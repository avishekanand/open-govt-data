#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Author gold SQL for verified (question, dataset) pairs, and prove it by running it.

Verification so far is a *reading of metadata*: a model looked at dimension names
and category values and judged the dataset adequate. That is an opinion. It can
be wrong in ways only execution exposes - in one case it named "Waarde" as the
measure when Waarde is a category of the `Marges` dimension (point estimate vs
confidence bounds), not a measure at all.

This pass closes that gap: the model is shown the REAL table schema from DuckDB
plus actual column values, writes one SELECT, and the query is executed. Only
queries that run and return a non-empty, sensibly-shaped result are kept, with
the answer pinned as a snapshot. One repair attempt is allowed, with the database
error fed back.

    python -m bench.author_sql --dry-run
    python -m bench.author_sql --model Qwen/Qwen3-32B
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bench.substrate import connect, view
from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

VERIFIED = Path("data/processed/benchmark/question_dataset_verified_cbs.jsonl")
OUT = Path("data/processed/benchmark/items_authored_cbs.jsonl")
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
MAX_ROWS = 200

SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
        "clarified_question": {"type": "string"},
        "assumptions": {"type": "string"},
    },
    "required": ["sql", "clarified_question", "assumptions"],
    "additionalProperties": False,
}

SYSTEM = ("You write DuckDB SQL over a single statistical table to answer a question. "
          "You use only columns and values that exist in the schema shown. "
          "Return STRICT JSON only.")

USER = """\
Question: {q}
Scope requested: {scope}

Table {code} — {title}
It is available as the view `{view}` with this schema:
{schema}

Actual values in each dimension column (sample):
{values}

Write ONE DuckDB SELECT that answers the question from `{view}`.

Rules:
- Use only columns and category values shown above.
- CBS tables hold aggregate rows next to their components: "Totaal ..." rows are
  totals. Never SUM across a dimension that contains its own total - filter to
  the total, or group by the components, never both.
- The `Perioden`/period column can mix granularities (years and weeks/months in
  one column). Filter to the granularity the question needs.
- Return a small result: a single row, or one row per group. Never dump raw rows.
- Round rates sensibly. Name output columns in English.

Also state:
- "clarified_question": the question with the choices your query makes spelled
  out (which period, which population, which measure).
- "assumptions": anything you had to decide that the question left open.
"""


def schema_of(con, vname: str) -> str:
    df = con.execute(f"DESCRIBE {vname}").df()
    return "\n".join(f"  {r.column_name} ({r.column_type})" for r in df.itertuples())


def values_of(con, vname: str, max_cols: int = 8, max_vals: int = 14) -> str:
    df = con.execute(f"DESCRIBE {vname}").df()
    out = []
    for r in df.itertuples():
        c = r.column_name
        if c in ("value", "value_text", "measure_code", "status"):
            continue
        try:
            vals = con.execute(
                f'SELECT DISTINCT "{c}" FROM {vname} WHERE "{c}" IS NOT NULL LIMIT {max_vals}'
            ).df()[c].astype(str).tolist()
        except Exception:  # noqa: BLE001
            continue
        if vals:
            out.append(f"  {c}: " + "; ".join(vals))
        if len(out) >= max_cols:
            break
    return "\n".join(out)


def qa_check(df) -> Optional[str]:
    """Reject results that ran but cannot be right.

    The commonest defect is an unfiltered dimension: CBS tables carry several
    dimensions, and forgetting one produces the same grouping key several times
    with different values. The query succeeds and the answer is meaningless.
    """
    import pandas as pd
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    keys = [c for c in df.columns if c not in num]
    if keys and df.duplicated(subset=keys).any():
        n = int(df.duplicated(subset=keys).sum())
        return (f"duplicate grouping keys ({n} rows): a dimension was not filtered, "
                f"so each key appears more than once")
    if not num:
        return None
    return None


def try_run(con, sql: str):
    try:
        df = con.execute(sql).df()
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    if df.empty:
        return None, "query returned no rows"
    if len(df) > MAX_ROWS:
        return None, f"query returned {len(df)} rows; expected an aggregated result"
    problem = qa_check(df)
    if problem:
        return None, problem
    return df, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Author and execute gold SQL")
    ap.add_argument("--verified", default=VERIFIED, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.verified.open(encoding="utf-8")]
    pairs = [r for r in rows if r.get("verified_dataset")]
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"[INFO] {len(pairs)} verified (question, dataset) pairs to author SQL for")

    con = connect()
    prompts, ctx = [], []
    for pi, r in enumerate(pairs, 1):
        print(f"[{pi}/{len(pairs)}] preparing {r.get('verified_dataset')}", flush=True)
        code = r["verified_dataset"]
        try:
            vname = view(con, code)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] skipping {code}: {exc}", flush=True)
            continue
        p = USER.format(
            q=r.get("question_selfcontained") or r.get("question"),
            scope=json.dumps({k: v for k, v in (r.get("scope") or {}).items() if v},
                             ensure_ascii=False) or "(not stated)",
            code=code, title=(r.get("verified_how") or {}).get("why", "")[:0] or code,
            view=vname, schema=schema_of(con, vname), values=values_of(con, vname))
        prompts.append(p)
        ctx.append((r, code, vname))
    print(f"[INFO] {len(prompts)} tables materialised")

    if args.dry_run:
        validate_schema_both_backends(SCHEMA, args.model)
        if prompts:
            print("\n--- sample prompt ---\n" + prompts[0][:1800])
        return
    if not prompts:
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)

    def gen(ps):
        convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}]
                 for p in ps]
        try:
            return llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            return llm.chat(convs, sampling)

    t0 = time.time()
    outs = gen(prompts)
    results: List[Optional[Dict[str, Any]]] = []
    repair_idx, repair_prompts = [], []
    for i, o in enumerate(outs):
        obj = parse_json(o.outputs[0].text if o.outputs else "")
        r, code, vname = ctx[i]
        if not obj or not obj.get("sql"):
            results.append(None)
            continue
        df, err = try_run(con, obj["sql"])
        if err:
            results.append({"obj": obj, "error": err})
            repair_idx.append(i)
            repair_prompts.append(prompts[i] +
                                  f"\n\nYour previous query failed:\n{obj['sql']}\n\n"
                                  f"Error: {err}\nFix it and return corrected JSON.")
        else:
            results.append({"obj": obj, "df": df})

    if repair_prompts:
        print(f"[INFO] repairing {len(repair_prompts)} failed queries")
        for i, o in zip(repair_idx, gen(repair_prompts)):
            obj = parse_json(o.outputs[0].text if o.outputs else "")
            if obj and obj.get("sql"):
                df, err = try_run(con, obj["sql"])
                results[i] = {"obj": obj, "df": df} if not err else {"obj": obj, "error": err}

    ok = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, res in enumerate(results):
            r, code, vname = ctx[i]
            if not res or res.get("df") is None:
                continue
            df = res["df"]
            ok += 1
            fh.write(json.dumps({
                "id": f"cbs-a{ok:04d}", "tier": "A_executable", "executable": True,
                "question_en": r.get("question_selfcontained") or r.get("question"),
                "question_verbatim": r.get("question"),
                "clarified_question": res["obj"].get("clarified_question"),
                "assumptions": res["obj"].get("assumptions"),
                "answer_type": r.get("answer_type"), "publisher": "CBS",
                "gold_datasets": [{"code": code, "publisher": "CBS"}],
                "gold_sql": res["obj"]["sql"], "sql_dialect": "duckdb",
                "answer_snapshot": {"columns": list(df.columns),
                                    "rows": df.astype(object).where(df.notna(), None).values.tolist(),
                                    "retrieved_at": datetime.date.today().isoformat()},
                "verification": "gold_sql executed successfully; result pinned",
                "source_publication": r.get("source_url"),
                "verified_how": r.get("verified_how"),
            }, ensure_ascii=False) + "\n")
    failed = sum(1 for x in results if not x or x.get("df") is None)
    print(f"[DONE] executed OK {ok} | failed {failed} -> {args.out} "
          f"in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
