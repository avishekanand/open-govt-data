#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Embedding retrieval from a question to the dataset catalogue.

Lexical linking hit a hard ceiling: it can only resolve a dataset when the
publication *cited it by name* and the wording happens to overlap. That lost 607
of 628 otherwise-usable questions - "weekly deaths" never reaches "Deceased
Persons by Gender and Age Group, per Week".

This drops the requirement that the paper cite anything. Each question is
embedded and matched against all 12,308 enriched datasets, so a dataset can be
*retrieved* rather than merely *resolved*. Output is a ranked candidate list per
question for human confirmation - it is retrieval evidence, not gold.

    python -m enrich.embed_link --top-k 10
    -> data/processed/benchmark/question_dataset_candidates.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

CORPUS = Path("data/processed/enriched_unified_qwen3-32b.jsonl")
SRC = Path("data/processed/pub/pub_evidence.jsonl")
OUT = Path("data/processed/benchmark/question_dataset_candidates.jsonl")
# bge-large-en-v1.5: CLS pooling, normalised; queries take an instruction prefix.
MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def dataset_text(rec: Dict[str, Any]) -> str:
    """What a dataset 'looks like' to the retriever."""
    parts = [rec.get("title_en") or "", rec.get("title_native") or ""]
    topics = rec.get("topics") or []
    if isinstance(topics, list):
        parts.append(", ".join(str(t) for t in topics))
    parts.append((rec.get("enriched_description") or "")[:400])
    dims = rec.get("dimensions") or []
    parts.append(", ".join(str(d.get("name") or d.get("id")) for d in dims[:8]))
    return " | ".join(p for p in parts if p)


def encode(texts: List[str], model, tok, device, batch_size: int, is_query: bool):
    import torch
    out = []
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, 0]        # CLS pooling
        out.append(torch.nn.functional.normalize(h, dim=-1).cpu())
        if (i // batch_size) % 20 == 0:
            print(f"  encoded {min(i + batch_size, len(texts)):,}/{len(texts):,}", flush=True)
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed questions and retrieve candidate datasets")
    ap.add_argument("--corpus", default=CORPUS, type=Path)
    ap.add_argument("--src", default=SRC, type=Path)
    ap.add_argument("--out", default=OUT, type=Path)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--only", default="public_aggregate",
                    help="restrict to a data_needed value; 'all' for everything")
    ap.add_argument("--publisher", default=None, help="restrict corpus to CBS or ESTAT")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    corpus = [json.loads(l) for l in args.corpus.open(encoding="utf-8")]
    if args.publisher:
        corpus = [c for c in corpus if c.get("publisher") == args.publisher]
    recs = [json.loads(l) for l in args.src.open(encoding="utf-8")]

    queries, meta = [], []
    for r in recs:
        for q in r.get("research_questions") or []:
            if args.only != "all" and q.get("data_needed") != args.only:
                continue
            text = q.get("question_selfcontained") or q.get("question_en") or q.get("question")
            if not text:
                continue
            queries.append(text)
            meta.append({
                "question": q.get("question_en"),
                "question_selfcontained": q.get("question_selfcontained"),
                "scope": q.get("scope"),
                "answer_type": q.get("answer_type"),
                "data_needed": q.get("data_needed"),
                "verifiable_now": q.get("verifiable_now"),
                "lexically_attributed": q.get("attributed_dataset"),
                "source_url": r.get("final_url") or r.get("url"),
                "source_title": r.get("title"),
            })
    print(f"[INFO] {len(queries):,} questions x {len(corpus):,} datasets | model={args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    print("[INFO] encoding catalogue...")
    demb = encode([dataset_text(c) for c in corpus], model, tok, device, args.batch_size, False)
    print("[INFO] encoding questions...")
    qemb = encode(queries, model, tok, device, args.batch_size, True)

    sims = qemb @ demb.T
    topv, topi = sims.topk(args.top_k, dim=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    hit_at_1 = hit_at_k = n_lex = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, m in enumerate(meta):
            cands = []
            for rank, (score, idx) in enumerate(zip(topv[i].tolist(), topi[i].tolist()), 1):
                c = corpus[idx]
                cands.append({"rank": rank, "score": round(float(score), 4),
                              "code": c.get("code"), "publisher": c.get("publisher"),
                              "title_en": c.get("title_en"),
                              "title_native": c.get("title_native")})
            # sanity signal: does retrieval recover the lexically-attributed dataset?
            lex = m.get("lexically_attributed")
            if lex:
                n_lex += 1
                codes = [c["code"] for c in cands]
                hit_at_1 += int(bool(codes) and codes[0].upper() == lex.upper())
                hit_at_k += int(any(c.upper() == lex.upper() for c in codes))
            fh.write(json.dumps({**m, "candidates": cands}, ensure_ascii=False) + "\n")

    print(f"[DONE] {len(meta):,} questions -> {args.out}")
    if n_lex:
        print(f"[CHECK] against {n_lex} lexically-attributed questions: "
              f"hit@1 {hit_at_1}/{n_lex}  hit@{args.top_k} {hit_at_k}/{n_lex}")


if __name__ == "__main__":
    main()
