#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch doc2query enrichment of CBS StatLine metadata with vLLM (HPC / GPU).

Server-free alternative to cbs.enrich_cbs: instead of talking to an Ollama
server, this loads a HuggingFace model directly with vLLM and runs ALL table
prompts through it as one batched job — the right shape for a SLURM GPU node.

vLLM downloads + caches the model from HF on first use (point HF_HOME at the
umbrella share to persist it). JSON output is enforced with guided decoding
when the installed vLLM version supports it, with a prompt + regex fallback.

Reads the same metadata as cbs.enrich_cbs:
    data/processed/catalog_meta/statline_{datasets,dimensions,measures}.parquet
Writes the same schema (one JSONL record per table).

Run locally on a GPU box or via scripts/enrich_daic.slurm:
    python -m cbs.enrich_cbs_vllm --model Qwen/Qwen2.5-7B-Instruct
    MODEL=google/gemma-2-9b-it python -m cbs.enrich_cbs_vllm   # gated: needs HF_TOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Reuse the prompt + parsing logic from the Ollama enricher (single source of truth).
from cbs.enrich_cbs import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    REQUIRED_KEYS,
    build_context,
    parse_json,
)

# JSON schema for guided decoding (keeps the same output contract).
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "title_en": {"type": "string"},
        "enriched_description": {"type": "string"},
        "example_queries": {"type": "array", "items": {"type": "string"}},
        "potential_applications": {"type": "array", "items": {"type": "string"}},
        "key_dimensions": {"type": "array", "items": {"type": "string"}},
        "topics": {"type": "array", "items": {"type": "string"}},
        "confidence": {
            "type": "object",
            "properties": {"desc": {"type": "number"}, "queries": {"type": "number"}},
        },
    },
    "required": sorted(REQUIRED_KEYS),
}

DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")


def load_metadata(meta_dir: Path):
    datasets = pd.read_parquet(meta_dir / "statline_datasets.parquet").drop_duplicates("table_id")
    dims = pd.read_parquet(meta_dir / "statline_dimensions.parquet")
    meas = pd.read_parquet(meta_dir / "statline_measures.parquet")
    return datasets, dims, meas


def already_done(output: Path) -> set:
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["code"])
            except Exception:
                pass
    return done


def build_prompts(datasets, dims, meas, done: set, limit) -> List[dict]:
    rows = datasets if not limit else datasets.head(limit)
    items = []
    for _, r in rows.iterrows():
        tid = str(r["table_id"])
        if tid in done:
            continue
        ctx = build_context(tid, dims, meas)
        user = USER_TEMPLATE.format(
            table_id=tid,
            title=r.get("title") or "",
            description=str(r.get("summary") or "")[:1500],
            dimensions=ctx["dimensions"],
            measures=ctx["measures"],
        )
        items.append({"table_id": tid, "title_nl": r.get("title"),
                      "source_url": r.get("source_url"), "user": user})
    return items


def make_sampling_params(temperature: float, max_tokens: int):
    from vllm import SamplingParams
    # Try guided JSON decoding (API name varies across vLLM versions).
    try:
        from vllm.sampling_params import GuidedDecodingParams
        gd = GuidedDecodingParams(json=JSON_SCHEMA)
        return SamplingParams(temperature=temperature, max_tokens=max_tokens, guided_decoding=gd)
    except Exception:
        try:
            return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                  guided_json=JSON_SCHEMA)  # older vLLM
        except Exception:
            print("[WARN] guided JSON unavailable; relying on prompt + regex parsing")
            return SamplingParams(temperature=temperature, max_tokens=max_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch CBS enrichment with vLLM")
    ap.add_argument("--meta-dir", default="data/processed/catalog_meta", type=Path)
    ap.add_argument("--output", default="data/processed/cbs_enriched_vllm.jsonl", type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="Enrich at most N tables")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--resume", action="store_true", help="Skip tables already in output")
    args = ap.parse_args()

    datasets, dims, meas = load_metadata(args.meta_dir)
    done = already_done(args.output) if args.resume else set()
    items = build_prompts(datasets, dims, meas, done, args.limit)
    print(f"[INFO] {len(items)} tables to enrich (skipped {len(done)} done) | model={args.model}")
    if not items:
        print("[DONE] nothing to do")
        return

    # Heavy imports happen only on the GPU node.
    from vllm import LLM
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    sampling = make_sampling_params(args.temperature, args.max_tokens)

    # One batched call — vLLM schedules all conversations across the GPU.
    conversations = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": it["user"]}]
        for it in items
    ]
    t0 = time.time()
    outputs = llm.chat(conversations, sampling)
    dt = time.time() - t0
    print(f"[INFO] generation done in {dt/60:.1f}m ({len(items)/max(dt,1):.1f} tables/s)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (args.resume and args.output.exists()) else "w"
    ok = fail = 0
    with args.output.open(mode, encoding="utf-8") as fout:
        for it, out in zip(items, outputs):
            text = out.outputs[0].text if out.outputs else ""
            obj = parse_json(text)
            if not obj or (REQUIRED_KEYS - set(obj.keys())):
                fail += 1
                continue
            obj["code"] = it["table_id"]
            obj["title_nl"] = it["title_nl"]
            obj["source_url"] = it["source_url"]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            ok += 1
    print(f"[DONE] enriched ok={ok} fail={fail} -> {args.output}")


if __name__ == "__main__":
    main()
