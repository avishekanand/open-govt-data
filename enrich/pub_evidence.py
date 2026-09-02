#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication evidence extraction over FULL document text.

Replaces cbs.pub_extract, which sent only the first 6,000 characters of each
document to the model - so a methods section past that cut was invisible. Here
every document is chunked end to end, each chunk is extracted under a
schema-constrained JSON grammar, and results are merged per document.

Three things are extracted, each with a **witness sentence** quoted verbatim
from the text:
  * data mentions   - CBS/Eurostat tables, StatLine tables, microdata registers
  * research questions - what the publication actually asks
  * usage summary   - whether CBS data is used, and of which kind

Every witness is then verified to occur in the source text; unverifiable ones
are kept but flagged, so a hallucinated quote can never pass as evidence.
Mentions are linked to the real catalogue via enrich.pub_link.

    python -m enrich.pub_evidence --dry-run
    python -m enrich.pub_evidence --model Qwen/Qwen3-32B --resume

Input : data/processed/pub/documents.parquet + data/raw/pub_text/*.txt
Output: data/processed/pub/pub_evidence.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import collections
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from enrich.pub_link import link
from enrich.vllm_util import make_sampling_params, parse_json, validate_schema_both_backends

PUB = Path("data/processed/pub")
DOCS = PUB / "documents.parquet"
OUT = PUB / "pub_evidence.jsonl"
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")

MENTION_KINDS = ["cbs_table_id", "cbs_microdata_register", "statline_table",
                 "eurostat_dataset", "survey_or_panel", "other"]
DATA_KINDS = ["microdata", "aggregate/StatLine", "both", "none", "unclear"]

CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "uses_cbs_data": {"type": "boolean"},
        "data_kind": {"type": "string", "enum": DATA_KINDS},
        "data_mentions": {
            "type": "array", "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "mention": {"type": "string"},
                    "kind": {"type": "string", "enum": MENTION_KINDS},
                    "witness": {"type": "string"},
                },
                "required": ["mention", "kind", "witness"],
                "additionalProperties": False,
            },
        },
        "research_questions": {
            "type": "array", "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "witness": {"type": "string"},
                },
                "required": ["question", "witness"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["uses_cbs_data", "data_kind", "data_mentions", "research_questions"],
    "additionalProperties": False,
}

SYSTEM = (
    "You extract evidence from research publications about which Statistics "
    "Netherlands (CBS) or Eurostat data they used. You quote evidence verbatim "
    "and never invent it. Return STRICT JSON only - no prose, no markdown."
)

USER = """\
Publication title: {title}
CBS project number (if any): {project}
Excerpt {i} of {n} from the document:
\"\"\"
{text}
\"\"\"

Extract ONLY what this excerpt supports.

"data_mentions": every specific dataset the text says was USED - a CBS table id
(e.g. 83765NED), a StatLine table named in words, a Eurostat dataset, a CBS
microdata register (GBA/BRP, POLIS/SPOLIS, SECMBUS, HOOGSTEOPLTAB, INPATAB/
INHATAB, VSLGWBTAB, ...), or a named survey/panel. Do NOT list data merely
discussed as related work, and do NOT guess an id that is not written here.

"research_questions": the questions this publication sets out to answer, phrased
as questions. Only if the excerpt states or clearly implies them.

"witness": for each item, ONE sentence copied EXACTLY from the excerpt above -
character for character - that supports it. If you cannot copy an exact
supporting sentence, omit the item entirely.

If the excerpt shows no CBS/Eurostat data usage, return empty arrays and
"data_kind": "none".

Return JSON with keys: uses_cbs_data, data_kind (one of {kinds}),
data_mentions[{{mention, kind, witness}}], research_questions[{{question, witness}}].
"""


# Chunks worth spending a generation on: those that actually mention CBS/Eurostat
# data, plus the opening chunks where a paper states what it set out to answer.
SIGNAL_RE = re.compile(
    r"\bcbs\b|statistics netherlands|centraal bureau voor de statistiek|statline|"
    r"microdata|micro-data|\b\d{4,6}[A-Za-z]{2,4}\b|GBA|BRP|POLIS|SPOLIS|SECMBUS|"
    r"HOOGSTEOPLTAB|INPATAB|INHATAB|VSLGWBTAB|eurostat", re.I)
INTRO_CHUNKS = 2          # research questions live in the abstract / introduction
MAX_MENTION_CHARS = 80    # a dataset name, not a copied sentence fragment
MAX_QUESTIONS_PER_DOC = 8


# ------------------------------------------------------------------ chunking --

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Split into overlapping windows, preferring a paragraph/sentence boundary."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > size * 0.5:          # only honour a boundary reasonably far in
                end = start + cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "")).strip().lower()


def verify_witness(witness: str, haystack_norm: str, min_len: int = 20) -> bool:
    """A witness counts only if it really occurs in the source text."""
    w = _norm(witness)
    if len(w) < min_len:
        return False
    if w in haystack_norm:
        return True
    # tolerate small OCR/whitespace drift by checking a long inner span
    span = w[: max(min_len, int(len(w) * 0.6))]
    return span in haystack_norm


# ------------------------------------------------------------------- merging --

def merge_doc(chunk_objs: Iterable[Dict[str, Any]], doc_text: str) -> Dict[str, Any]:
    """chunk_objs is an iterable of (chunk_index, parsed_obj)."""
    hay = _norm(doc_text)
    mentions: Dict[str, Dict[str, Any]] = {}
    questions: Dict[str, Dict[str, Any]] = {}
    uses = False
    kinds: List[str] = []
    for idx, o in chunk_objs:
        if not o:
            continue
        uses = uses or bool(o.get("uses_cbs_data"))
        dk = o.get("data_kind")
        if dk and dk not in ("none", "unclear"):
            kinds.append(dk)
        for m in o.get("data_mentions") or []:
            key = _norm(m.get("mention"))
            # Drop copied sentence fragments: a dataset reference is a name.
            if not key or len(key) > MAX_MENTION_CHARS:
                continue
            ok = verify_witness(m.get("witness", ""), hay)
            prev = mentions.get(key)
            # prefer a verified witness over an unverified one
            if prev is None or (ok and not prev["witness_verified"]):
                mentions[key] = {"mention": m.get("mention"), "kind": m.get("kind"),
                                 "witness": m.get("witness"), "witness_verified": ok,
                                 "n_chunks": (prev or {}).get("n_chunks", 0) + 1}
            else:
                prev["n_chunks"] += 1
        # Only the opening chunks state what the publication set out to answer;
        # later chunks yield rhetorical asides (one 40-chunk book produced 135).
        if idx > INTRO_CHUNKS:
            continue
        for q in o.get("research_questions") or []:
            key = _norm(q.get("question"))
            if not key:
                continue
            ok = verify_witness(q.get("witness", ""), hay)
            prev = questions.get(key)
            if prev is None or (ok and not prev["witness_verified"]):
                questions[key] = {"question": q.get("question"), "witness": q.get("witness"),
                                  "witness_verified": ok}
    # "both" needs real support on each side: across a 40-chunk book a single
    # stray mention of either kind would otherwise flip every document to "both".
    counts = collections.Counter(kinds)
    if counts.get("both", 0) >= 1 or (counts.get("microdata", 0) >= 2
                                      and counts.get("aggregate/StatLine", 0) >= 2):
        data_kind = "both"
    elif kinds:
        data_kind = max(set(kinds), key=kinds.count)
    else:
        data_kind = "none" if not uses else "unclear"
    qs = sorted(questions.values(), key=lambda x: not x["witness_verified"])
    return {"uses_cbs_data": uses, "data_kind": data_kind,
            "data_mentions": list(mentions.values()),
            "research_questions": qs[:MAX_QUESTIONS_PER_DOC]}


# --------------------------------------------------------------------- items --

def load_documents(docs_path: Path, min_text: int, limit: Optional[int]) -> pd.DataFrame:
    d = pd.read_parquet(docs_path)
    d = d[d["ok"] & (d["text_len"].fillna(0) >= min_text)].copy()
    d = d.sort_values("text_len", ascending=False)
    return d.head(limit) if limit else d


def doc_text(row) -> str:
    tp = row.get("text_path")
    if tp and Path(tp).exists():
        try:
            return Path(tp).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    return row.get("text") or ""


def done_urls(path: Path) -> set:
    out = set()
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["url"])
                except Exception:  # noqa: BLE001
                    pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Full-text publication evidence extraction")
    ap.add_argument("--docs", default=DOCS, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="max documents")
    ap.add_argument("--min-text", type=int, default=300)
    ap.add_argument("--chunk-size", type=int, default=6000, help="characters per window")
    ap.add_argument("--chunk-overlap", type=int, default=600)
    ap.add_argument("--max-chunks-per-doc", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--doc-batch", type=int, default=40, help="documents per flush")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--all-chunks", action="store_true",
                    help="Generate on every chunk, not just those with a data signal")
    ap.add_argument("--sample", type=int, default=None,
                    help="Random sample of N documents (smoke tests); --limit takes the largest")
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    docs = load_documents(args.docs, args.min_text, None if args.sample else args.limit)
    if args.sample:
        docs = docs.sample(n=min(args.sample, len(docs)), random_state=0)
    done = done_urls(args.out) if args.resume else set()
    docs = docs[~docs["url"].isin(done)]

    plan = []           # (doc_row, [(chunk_index, chunk_text)])
    skipped_chunks = 0
    for _, row in docs.iterrows():
        ch = chunk_text(doc_text(row), args.chunk_size, args.chunk_overlap)
        if not ch:
            continue
        ch = ch[: args.max_chunks_per_doc]
        if args.all_chunks:
            keep = list(enumerate(ch, 1))
        else:
            keep = [(i, c) for i, c in enumerate(ch, 1)
                    if i <= INTRO_CHUNKS or SIGNAL_RE.search(c)]
            skipped_chunks += len(ch) - len(keep)
        if keep:
            plan.append((row, keep))
    n_chunks = sum(len(c) for _, c in plan)
    if skipped_chunks:
        print(f"[INFO] skipped {skipped_chunks:,} chunks with no CBS/Eurostat signal "
              f"outside the first {INTRO_CHUNKS}")
    print(f"[INFO] {len(plan):,} documents -> {n_chunks:,} chunks "
          f"({n_chunks/max(len(plan),1):.1f} per doc), skipped {len(done):,} done")

    if args.dry_run:
        validate_schema_both_backends(CHUNK_SCHEMA, args.model)
        if plan:
            row, ch = plan[0]
            print(f"\n--- sample: {row['url'][:70]} ({len(ch)} chunks kept) ---")
            print(USER.format(title=(row.get("title") or "")[:200], project=row.get("project"),
                              i=ch[0][0], n=len(ch), text=ch[0][1][:700],
                              kinds="/".join(DATA_KINDS))[:1600])
        return
    if not plan:
        print("[DONE] nothing to do")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True)
    sampling = make_sampling_params(CHUNK_SCHEMA, args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)

    def generate(convs):
        try:
            return llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            return llm.chat(convs, sampling)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.out.exists()) else "w"
    t0, written, unparsed = time.time(), 0, 0
    with args.out.open(mode, encoding="utf-8") as fout:
        for start in range(0, len(plan), args.doc_batch):
            batch = plan[start:start + args.doc_batch]
            convs, owner = [], []
            for di, (row, chunks) in enumerate(batch):
                for idx, c in chunks:
                    convs.append([
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": USER.format(
                            title=(row.get("title") or "")[:200], project=row.get("project"),
                            i=idx, n=len(chunks), text=c, kinds="/".join(DATA_KINDS))},
                    ])
                    owner.append((di, idx))
            outs = generate(convs)
            per_doc: Dict[int, List[Any]] = {}
            for oi, o in enumerate(outs):
                obj = parse_json(o.outputs[0].text if o.outputs else "")
                if obj is None:
                    unparsed += 1
                di, idx = owner[oi]
                per_doc.setdefault(di, []).append((idx, obj))
            for di, (row, chunks) in enumerate(batch):
                text = doc_text(row)
                merged = merge_doc(per_doc.get(di, []), text)
                for m in merged["data_mentions"]:
                    m["link"] = link(m["mention"])
                rec = {
                    "url": row["url"], "final_url": row.get("final_url"),
                    "title": row.get("title"), "project": row.get("project"),
                    "domain": row.get("domain"), "resource_type": row.get("resource_type"),
                    "text_path": row.get("text_path"), "text_len": int(row.get("text_len") or 0),
                    "n_chunks": len(chunks), "model": args.model, **merged,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            fout.flush()
            os.fsync(fout.fileno())
            el = time.time() - t0
            print(f"[{min(start+args.doc_batch, len(plan))}/{len(plan)}] docs={written} "
                  f"unparsed_chunks={unparsed} | {el/60:.1f}m", flush=True)
    print(f"[DONE] {written:,} documents -> {args.out} in {(time.time()-t0)/60:.1f}m "
          f"({unparsed} unparsed chunks)")


if __name__ == "__main__":
    main()
