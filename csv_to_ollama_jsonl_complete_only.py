#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csv_to_ollama_jsonl_loose_verbose.py

Read Eurostat-like metadata from CSV, skip incomplete rows, build prompts,
call a local Ollama model (e.g., gemma3:9b-instruct), parse either:
  - strict JSON (if the model returns it), OR
  - loose sections (headings/bullets) and synthesize the final object,
validate against a schema, and write JSONL.

Verbose, colorized terminal output with progress and error reporting.

Usage (loose, recommended):
  python csv_to_ollama_jsonl_loose_verbose.py \
      --input eurostat_enriched.csv \
      --output eurostat_gemma3.jsonl \
      --model gemma3:9b-instruct \
      --temperature 0.2 \
      --prompt-mode loose \
      --show-output

Usage (strict JSON mode, if you want to try it):
  python csv_to_ollama_jsonl_loose_verbose.py \
      --input eurostat_enriched.csv \
      --output eurostat_gemma3.jsonl \
      --model gemma3:9b-instruct \
      --temperature 0.2 \
      --prompt-mode json

Requirements:
- Python 3.9+
- pip install requests
- Ollama running locally with your chosen model pulled (e.g., gemma3:9b-instruct)
"""

import argparse
import ast
import csv
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

import requests  # pip install requests

# ========= Terminal styling (no external deps) =========

class Style:
    def __init__(self):
        self.enabled = sys.stdout.isatty()
        if self.enabled:
            self.RESET   = "\033[0m"
            self.BOLD    = "\033[1m"
            self.DIM     = "\033[2m"
            self.ITALIC  = "\033[3m"
            self.UNDER   = "\033[4m"
            self.GRAY    = "\033[90m"
            self.RED     = "\033[91m"
            self.GREEN   = "\033[92m"
            self.YELLOW  = "\033[93m"
            self.BLUE    = "\033[94m"
            self.MAGENTA = "\033[95m"
            self.CYAN    = "\033[96m"
        else:
            self.RESET = self.BOLD = self.DIM = self.ITALIC = self.UNDER = ""
            self.GRAY = self.RED = self.GREEN = self.YELLOW = self.BLUE = self.MAGENTA = self.CYAN = ""
S = Style()

def hr(char="─", width=80):
    return char * width

def truncate(s: str, n=220):
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"

def box(title: str, body: str, width=80):
    title_line = f"{S.BOLD}{title}{S.RESET}"
    top = "┌" + "─" * (width - 2) + "┐"
    bot = "└" + "─" * (width - 2) + "┘"
    lines = [top, f"│ {title_line.ljust(width-4)} │"]
    for line in body.splitlines() if body else []:
        for chunk_start in range(0, len(line), width - 4):
            chunk = line[chunk_start:chunk_start + (width - 4)]
            lines.append(f"│ {chunk.ljust(width-4)} │")
    lines.append(bot)
    return "\n".join(lines)

# ========= Prompts =========

SYSTEM_PROMPT_BASE = """You are a structured data-to-text generator for open statistical datasets.
Be concise, neutral, and avoid causal claims. Mention how measures are broken down using given dimensions.
"""

# Strict JSON mode (optional)
SYSTEM_PROMPT_JSON = SYSTEM_PROMPT_BASE + """
Return ONLY raw JSON (no code fences, no markdown, no pre/post text).
"""

USER_TEMPLATE_JSON = """INPUT COLUMNS:
title | code | type | lastupdateofdata | lasttablestructurechange | datastart | dataend | label | source | updated | dimensions

ROW:
{title}
{code}
{type}
{lastupdateofdata}
{lasttablestructurechange}
{datastart}
{dataend}
{label}
{source}
{updated}
{dimensions}

TASK:
Produce exactly one JSON object with keys:
code,title,years({{"start":int,"end":int}}),source,last_update,dimensions,enriched_description(<=70 words),
example_queries(6 QUESTION strings - each must be a complete question ending with ?),potential_applications(3-5 strings),join_keys,notes,confidence({{"desc":float,"queries":float,"apps":float}}).

For example_queries, generate 6 specific analytical questions about this dataset, such as:
- "How has [measure] changed over time from [start] to [end]?"
- "Which countries show the highest rates of [measure]?"
- "What are the differences in [measure] between men and women?"
- "How does [measure] vary by age group?"
- "Which geographic regions are outliers in [measure]?"
- "What is the data coverage for [measure] across all dimensions?"

Output ONLY raw JSON.
"""

# Loose mode (recommended)
USER_TEMPLATE_LOOSE = """INPUT DATASET:
Title: {title}
Code: {code}
Type: {type}
Last Update: {lastupdateofdata}
Last Structure Change: {lasttablestructurechange}
Data Start: {datastart}
Data End: {dataend}
Label: {label}
Source: {source}
Updated: {updated}
Dimensions: {dimensions}

Please summarize this dataset using the following SIMPLE TEXT format (no JSON required):

Enriched description:
<1 short paragraph, ≤70 words. Mention how measures are broken down using given dimensions.>

Example queries:
- What trends are visible in {title} from {datastart} to {dataend}?
- How does {title} vary across different geographic regions?
- Which age groups show the highest rates of {title}?
- What are the gender differences in {title} patterns?
- Which countries are outliers in {title} measurements?
- How complete is the {title} data across all dimensions?

Potential applications:
- <3–5 practical public-sector uses with likely users/outputs.>

Join keys:
<comma-separated subset from: geo, time, sex, age, education, sector, region>

Notes:
<optional clarifications.>

If unsure about measure type, say "rates or counts depending on unit".
Keep to these sections in this exact order, and be brief.
"""
REQUIRED_INPUT_COLS = [
    "title", "code", "datastart", "dataend", "dimensions", "updated"
]

REQUIRED_OUTPUT_KEYS = {
    "code", "title", "years", "source", "last_update", "dimensions",
    "enriched_description", "example_queries", "potential_applications",
    "join_keys", "confidence"
}

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

def parse_dimensions(raw: str) -> List[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return [str(x).strip() for x in val]
    except Exception:
        pass
    cleaned = raw.strip("[]")
    parts = [p.strip().strip("'").strip('"') for p in cleaned.split(",")]
    return [p for p in parts if p]

def normalize_row(row: Dict[str, str]) -> Dict[str, Any]:
    keys = [
        "title", "code", "type", "lastupdateofdata", "lasttablestructurechange",
        "datastart", "dataend", "label", "source", "updated", "dimensions"
    ]
    out = {k: (row.get(k, "") if row.get(k, "") is not None else "").strip() for k in keys}
    for k in ("datastart", "dataend"):
        try:
            out[k] = int(out[k]) if out[k] else None
        except Exception:
            out[k] = None
    out["_dims_list"] = parse_dimensions(out.get("dimensions", ""))
    return out

def row_is_complete(row: Dict[str, Any]) -> bool:
    for c in REQUIRED_INPUT_COLS:
        val = str(row.get(c, "")).strip()
        if not val or val.lower() == "nan":
            return False
    return True

def row_hash(nrow: Dict[str, Any]) -> str:
    data = {k: v for k, v in nrow.items() if k != "_dims_list"}
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def build_user_prompt(nrow: Dict[str, Any], mode: str) -> str:
    if mode == "json":
        tmpl = USER_TEMPLATE_JSON
    else:
        tmpl = USER_TEMPLATE_LOOSE
    return tmpl.format(
        title=nrow.get("title", ""),
        code=nrow.get("code", ""),
        type=nrow.get("type", ""),
        lastupdateofdata=nrow.get("lastupdateofdata", ""),
        lasttablestructurechange=nrow.get("lasttablestructurechange", ""),
        datastart=(nrow.get("datastart") if nrow.get("datastart") is not None else ""),
        dataend=(nrow.get("dataend") if nrow.get("dataend") is not None else ""),
        label=nrow.get("label", ""),
        source=nrow.get("source", ""),
        updated=nrow.get("updated", ""),
        dimensions=nrow.get("dimensions", ""),
    )

# ========= Ollama call =========

def call_ollama_chat(
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    num_ctx: int = 4096,
    retries: int = 3,
    timeout: int = 120,
    format_json: bool = False,   # default False for "no forcing"
) -> str:
    """
    Call Ollama's /api/chat endpoint and return assistant content (string).
    If format_json=True, include "format":"json" hint (still not strictly enforced).
    """
    url = host.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature, "num_ctx": num_ctx},
        "stream": False,
    }
    if format_json:
        payload["format"] = "json"  # hint, not enforced if you don't want to
    backoff = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            content = data.get("message", {}).get("content", "") or data.get("content", "")
            if not content:
                raise RuntimeError(f"Ollama returned empty content: {data}")
            return content
        except Exception as e:
            if attempt >= retries:
                raise
            print(f"{S.YELLOW}  ↻ Retry {attempt}/{retries} after error: {e}{S.RESET}")
            time.sleep(backoff)
            backoff *= 1.8

# ========= JSON extraction (robust) =========

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(text: str) -> str:
    return FENCE_RE.sub("", text).strip()

def _first_balanced_block(text: str) -> Optional[str]:
    s = text
    start = s.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def _coerce_json_like(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        pyobj = ast.literal_eval(s)
        if isinstance(pyobj, dict):
            return json.loads(json.dumps(pyobj, ensure_ascii=False))
    except Exception:
        pass
    return None

def extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    raw = text.strip()

    # Direct strict
    try:
        json.loads(raw)
        return raw
    except Exception:
        pass

    # Strip code fences
    stripped = _strip_code_fences(raw)
    if stripped and stripped != raw:
        try:
            json.loads(stripped)
            print(f"{S.GRAY}    • Stripped code fences to parse JSON.{S.RESET}")
            return stripped
        except Exception:
            pass

    # First balanced block
    block = _first_balanced_block(stripped or raw)
    if block:
        try:
            json.loads(block)
            print(f"{S.GRAY}    • Parsed first balanced JSON block.{S.RESET}")
            return block
        except Exception:
            coerced = _coerce_json_like(block)
            if coerced is not None:
                print(f"{S.GRAY}    • Coerced JSON-like block via ast.literal_eval.{S.RESET}")
                return json.dumps(coerced, ensure_ascii=False)

    # Last resort: coerce whole text
    coerced = _coerce_json_like(stripped or raw)
    if coerced is not None:
        print(f"{S.GRAY}    • Coerced full response via ast.literal_eval.{S.RESET}")
        return json.dumps(coerced, ensure_ascii=False)

    return None

# ========= Loose-format parsing & synthesis =========

SECTION_NAMES = ["enriched description", "example queries", "potential applications", "join keys", "notes"]
ALLOWED_JOIN = {"geo", "time", "sex", "age", "education", "sector", "region"}

def strip_code_fences_simple(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        i = t.find("\n")
        if i != -1:
            t = t[i+1:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()

def split_sections_loose(text: str) -> dict:
    t = strip_code_fences_simple(text)
    lines = [ln.rstrip() for ln in t.splitlines()]
    buf, current = {}, None
    for ln in lines:
        low = ln.strip().lower()
        matched = False
        for name in SECTION_NAMES:
            if low.startswith(name) and (low == name or low.startswith(name + ":")):
                current = name
                buf.setdefault(current, [])
                after = ln.split(":", 1)[1].strip() if ":" in ln else ""
                if after:
                    buf[current].append(after)
                matched = True
                break
        if not matched and current is not None:
            buf[current].append(ln)
    return {k: "\n".join(v).strip() for k, v in buf.items()}

def parse_bullets(text: str) -> list:
    if not text:
        return []
    items = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s[0] in "-*•":
            s = s[1:].strip()
        items.append(s)
    return items

def sanitize_join_keys(s: str, dims_from_csv: list) -> list:
    items = [x.strip().lower() for x in s.replace(";", ",").split(",") if x.strip()]
    picks = []
    for it in items:
        if it in ALLOWED_JOIN and (not dims_from_csv or it in dims_from_csv or it in {"geo","time","sex","age"}):
            picks.append(it)
    if picks:
        return list(dict.fromkeys(picks))
    fallback = [x for x in ["geo", "time", "sex", "age"] if x in dims_from_csv]
    return fallback

def clamp_words(s: str, max_words: int) -> str:
    ws = s.split()
    return " ".join(ws[:max_words]) if len(ws) > max_words else s

def ensure_six_queries(queries: list, dims: list, years: tuple) -> list:
    q = [clamp_words(x, 18) for x in queries if x.strip()][:6]
    
    # Only add fallbacks if we have very few queries (< 3)
    if len(q) >= 3:
        # Pad to exactly 6 with simple variations if needed
        while len(q) < 6:
            if len(q) == 3:
                q.append("What are the data quality issues or gaps?")
            elif len(q) == 4:
                q.append("How has data collection methodology changed over time?")
            elif len(q) == 5:
                q.append("What external factors might influence these measures?")
        return q[:6]
    
    # Only use generic fallbacks if we have < 3 specific queries
    start, end = years
    yr_span = f"{start}–{end}" if start and end else "the available period"
    has_geo = "geo" in dims
    has_sex = "sex" in dims
    has_age = "age" in dims
    
    templates = [
        f"How did measures trend over {yr_span}?",
        "Which geographic areas show the highest values?" if has_geo else "Which categories show the highest values?",
        "How do results differ by demographic groups?" if (has_sex or has_age) else "How do subgroup results vary?",
        "What are the most significant changes year-over-year?",
        "Which data points appear to be outliers?",
        "What is the overall data coverage and completeness?"
    ]
    
    # Fill remaining slots with templates
    while len(q) < 6 and len(templates) > 0:
        q.append(templates.pop(0))
    
    return q[:6]

def ensure_apps(apps: list) -> list:
    a = [clamp_words(x, 24) for x in apps if x][:5]
    if len(a) >= 3:
        return a
    extras = [
        "Policy brief with cross-country benchmarks.",
        "Monitoring dashboard for ministries and statistical offices.",
        "Coverage quality checks and data completeness alerts."
    ]
    for e in extras:
        if len(a) >= 3:
            break
        a.append(e)
    return a[:5]

def build_schema_from_sections(secs: dict, csv_row: dict) -> dict:
    desc = secs.get("enriched description", "").strip()
    queries = parse_bullets(secs.get("example queries", ""))
    apps = parse_bullets(secs.get("potential applications", ""))
    join = sanitize_join_keys(secs.get("join keys", ""), parse_dimensions(csv_row.get("dimensions","")))
    notes = secs.get("notes", "").strip()

    dims_list = parse_dimensions(csv_row.get("dimensions", ""))
    start = csv_row.get("datastart")
    end = csv_row.get("dataend")
    years = (
        int(start) if str(start).isdigit() else None,
        int(end) if str(end).isdigit() else None
    )

    obj = {
        "code": csv_row.get("code",""),
        "title": csv_row.get("label") or csv_row.get("title",""),
        "years": {"start": years[0], "end": years[1]},
        "source": csv_row.get("source","") or "ESTAT",
        "last_update": csv_row.get("updated","") or csv_row.get("lastupdateofdata",""),
        "dimensions": dims_list,
        "enriched_description": clamp_words(desc, 70),
        "example_queries": ensure_six_queries(queries, dims_list, years),
        "potential_applications": ensure_apps(apps),
        "join_keys": join,
        "notes": notes,
        "confidence": {"desc": 0.85, "queries": 0.9, "apps": 0.85}
    }
    return obj

# ========= Validation =========

def validate_output(obj: Dict[str, Any]) -> List[str]:
    problems = []
    missing = REQUIRED_OUTPUT_KEYS - set(obj.keys())
    if missing:
        problems.append(f"Missing keys: {sorted(missing)}")
    if not isinstance(obj.get("example_queries", None), list) or len(obj["example_queries"]) != 6:
        problems.append("example_queries must be a list of 6 items.")
    apps = obj.get("potential_applications", None)
    if not isinstance(apps, list) or not (3 <= len(apps) <= 5):
        problems.append("potential_applications must be a list of 3–5 items.")
    desc = obj.get("enriched_description", "")
    if isinstance(desc, str) and len(desc.split()) > 70:
        problems.append("enriched_description exceeds 70 words.")
    return problems

# ========= Progress bar =========

def progress_line(done, total, cached, skipped, failed, width=80):
    bar_w = max(10, min(40, width - 40))
    pct = 0 if total == 0 else int(done * 100 / total)
    filled = int(bar_w * pct / 100)
    bar = f"[{S.GREEN}{'█'*filled}{S.RESET}{'░'*(bar_w-filled)}]"
    return (f" {bar} {pct:3d}% "
            f"{S.CYAN}done:{done}{S.RESET} "
            f"{S.YELLOW}cached:{cached}{S.RESET} "
            f"{S.GRAY}skipped:{skipped}{S.RESET} "
            f"{S.RED}failed:{failed}{S.RESET}")

# ========= Main =========

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV.")
    ap.add_argument("--output", required=True, help="Path to output JSONL.")
    ap.add_argument("--model", default="gemma3:9b-instruct", help="Ollama model name.")
    ap.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                    help="Ollama host (default: http://localhost:11434 or $OLLAMA_HOST).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=0.15, help="Sleep between rows to reduce CPU spikes.")
    ap.add_argument("--cache", default="ollama_cache.json", help="Path to cache JSON.")
    ap.add_argument("--show-output", action="store_true", help="Pretty-print the JSON returned per row.")
    ap.add_argument("--max-rows", type=int, default=None, help="Process at most N rows.")
    ap.add_argument("--prompt-mode", choices=["loose", "json"], default="loose",
                    help="Use 'loose' sections (post-processed) or 'json' strict mode.")
    ap.add_argument("--format-json-hint", action="store_true",
                    help="Send format='json' hint to Ollama (kept off by default).")
    args = ap.parse_args()

    print(box("CSV → Ollama JSON Enrichment",
              f"Model: {args.model}\nHost:  {args.host}\nInput: {args.input}\nOutput: {args.output}\nCache:  {args.cache}\n"
              f"Temp:  {args.temperature} | Ctx: {args.num_ctx} | Retries: {args.retries}\n"
              f"Prompt mode: {args.prompt_mode} | format_json_hint: {args.format_json_hint}"))

    # Pick system prompt based on mode
    system_prompt = SYSTEM_PROMPT_JSON if args.prompt_mode == "json" else SYSTEM_PROMPT_BASE

    # Load cache
    if os.path.exists(args.cache):
        try:
            with open(args.cache, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    else:
        cache = {}
    seen_hashes = set(cache.keys())

    # Open output
    out_mode = "a" if os.path.exists(args.output) else "w"
    out_f = open(args.output, out_mode, encoding="utf-8")

    # Pre-count rows for progress
    try:
        with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
            total_rows = sum(1 for _ in csv.DictReader(f))
    except Exception:
        total_rows = 0

    total = 0
    skipped = 0
    cached = 0
    generated = 0
    failed = 0

    with open(args.input, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row_idx, row in enumerate(reader, start=1):
            if args.max_rows is not None and row_idx > args.max_rows:
                break

            total += 1
            row.pop("Unnamed: 0", None)  # optional index col

            print("\n" + hr())
            header = f"{S.BOLD}Row {row_idx}{S.RESET}  code={S.BLUE}{row.get('code','').strip()}{S.RESET}  title={truncate(row.get('title',''))}"
            print(header)

            if not row_is_complete(row):
                skipped += 1
                print(f"{S.GRAY}  • Skipped: incomplete required fields (title/code/datastart/dataend/dimensions).{S.RESET}")
                print(progress_line(cached + generated + failed + skipped, total_rows or total, cached, skipped, failed))
                continue

            nrow = normalize_row(row)
            if nrow.get("datastart") is None or nrow.get("dataend") is None:
                skipped += 1
                print(f"{S.GRAY}  • Skipped: datastart/dataend not parseable as integers.{S.RESET}")
                print(progress_line(cached + generated + failed + skipped, total_rows or total, cached, skipped, failed))
                continue

            h = row_hash(nrow)
            if h in seen_hashes:
                cached += 1
                out_f.write(json.dumps(cache[h], ensure_ascii=False) + "\n")
                print(f"{S.YELLOW}  • Cached hit. Written cached JSONL line.{S.RESET}")
                print(progress_line(cached + generated + failed + skipped, total_rows or total, cached, skipped, failed))
                time.sleep(args.sleep)
                continue

            user_prompt = build_user_prompt(nrow, args.prompt_mode)
            print(f"{S.DIM}  • Prompt preview ({args.prompt_mode}):{S.RESET}")
            print(truncate(user_prompt, 500))

            try:
                # Call Ollama
                print(f"{S.CYAN}  • Calling Ollama…{S.RESET}")
                resp_text = call_ollama_chat(
                    host=args.host,
                    model=args.model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=args.temperature,
                    num_ctx=args.num_ctx,
                    retries=args.retries,
                    format_json=(args.prompt_mode == "json" and args.format_json_hint),
                )
                print(f"{S.CYAN}    ↳ Response length: {len(resp_text)} chars{S.RESET}")

                # Try strict JSON first (even in loose mode — sometimes models oblige)
                obj = None
                json_block = extract_json_block(resp_text)
                if json_block:
                    obj = json.loads(json_block)
                    print(f"{S.GRAY}    • Parsed strict JSON from model output.{S.RESET}")
                else:
                    # Loose mode parsing
                    secs = split_sections_loose(resp_text)
                    if not secs:
                        raise ValueError("Could not extract JSON or recognizable sections from model output.")
                    obj = build_schema_from_sections(secs, nrow)
                    print(f"{S.GRAY}    • Parsed simple sections and built schema.{S.RESET}")

                # Validate and annotate
                problems = validate_output(obj)
                if problems:
                    diag = " | ".join(problems)
                    obj.setdefault("notes", "")
                    obj["notes"] = (obj["notes"] + " " + f"[validator] {diag}").strip()
                    print(f"{S.YELLOW}  • Validator notes: {diag}{S.RESET}")
                else:
                    print(f"{S.GREEN}  • JSON validates against schema constraints.{S.RESET}")

                # Show pretty JSON (optional)
                if args.show_output:
                    print(f"{S.BOLD}{S.BLUE}  • Output JSON:{S.RESET}")
                    print(json.dumps(obj, ensure_ascii=False, indent=2))

                # Persist
                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                cache[h] = obj
                generated += 1

            except Exception as e:
                failed += 1
                print(f"{S.RED}  ✖ Error: {e}{S.RESET}")
                if 'resp_text' in locals():
                    print(f"{S.RED}{S.DIM}    Raw response excerpt:{S.RESET} {truncate(resp_text, 400)}")
            finally:
                print(progress_line(cached + generated + failed + skipped, total_rows or total, cached, skipped, failed))
                time.sleep(args.sleep)

    out_f.close()
    # Save cache
    tmp = args.cache + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.cache)

    # ========= Summary =========
    print("\n" + hr())
    print(box("Run Summary",
              f"Input:     {args.input}\n"
              f"Output:    {args.output}\n"
              f"Model:     {args.model}\n"
              f"Host:      {args.host}\n"
              f"Cache:     {args.cache}\n\n"
              f"Processed: {total}\n"
              f"Generated: {generated}\n"
              f"Cached:    {cached}\n"
              f"Skipped:   {skipped}\n"
              f"Failed:    {failed}\n"))
    if failed:
        print(f"{S.RED}{S.BOLD}Some rows failed. See errors above for details.{S.RESET}")
    else:
        print(f"{S.GREEN}{S.BOLD}All done successfully!{S.RESET}")

if __name__ == "__main__":
    main()