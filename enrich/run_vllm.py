#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified batched enrichment for CBS StatLine + Eurostat via vLLM.

One model, one schema, one output file - records carry `publisher` so both
sources can share an index.

    python -m enrich.run_vllm --source both --model Qwen/Qwen3-32B
    python -m enrich.run_vllm --source eurostat --limit 20 --model Qwen/Qwen3-32B

JSON is schema-constrained (xgrammar). If the installed vLLM exposes no
structured-output API the run aborts rather than silently free-forming; pass
--allow-unconstrained to override.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from enrich.schema import (
    LLM_JSON_SCHEMA, LLM_REQUIRED_KEYS, SYSTEM_PROMPT, USER_TEMPLATE, finalize_record,
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
DEFAULT_MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    # Reasoning models may emit a <think> block before the JSON; drop it.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None


def validate_schema_both_backends(model: str) -> None:
    """Compile the schema with BOTH structured-output backends.

    vLLM chooses the backend itself (backend='auto'). xgrammar and llguidance
    support different JSON Schema subsets - `uniqueItems` compiles under
    xgrammar and hard-fails the whole job under llguidance - so validating
    against only one is how a schema change reaches the GPU and dies there.
    """
    import json as _json
    try:
        import xgrammar
        xgrammar.Grammar.from_json_schema(_json.dumps(LLM_JSON_SCHEMA))
        print("[SCHEMA] xgrammar   : OK")
    except Exception as exc:  # noqa: BLE001
        print(f"[SCHEMA] xgrammar   : FAIL -> {exc}")
        raise
    try:
        from transformers import AutoTokenizer
        from vllm.sampling_params import (SamplingParams, StructuredOutputsParams,
                                          _get_llg_tokenizer)
        from vllm.v1.structured_output.backend_guidance import validate_guidance_grammar
        tok = AutoTokenizer.from_pretrained(model)
        validate_guidance_grammar(
            SamplingParams(structured_outputs=StructuredOutputsParams(json=LLM_JSON_SCHEMA)),
            tokenizer=_get_llg_tokenizer(tok))
        print("[SCHEMA] llguidance : OK")
    except ImportError as exc:
        print(f"[SCHEMA] llguidance : SKIPPED (vLLM internals moved: {exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"[SCHEMA] llguidance : FAIL -> {exc}")
        raise


def collect_items(source: str, limit: Optional[int], require_metadata: bool) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if source in ("cbs", "both"):
        from enrich.context_cbs import iter_items as cbs_items
        items += list(cbs_items(limit=limit))
    if source in ("eurostat", "both"):
        from enrich.context_eurostat import iter_items as est_items
        # Enrich what has harvested structure; the rest are picked up by a later
        # --resume run once their metadata is ingested, rather than being burned
        # into the output as ungrounded records now.
        items += list(est_items(limit=limit, require_metadata=require_metadata))
    return items


def already_done(output: Path) -> set:
    done = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                done.add((rec.get("publisher"), rec.get("code")))
            except Exception:  # noqa: BLE001
                pass
    return done


def make_sampling_params(temperature: float, max_tokens: int, require_schema: bool = True):
    """JSON-schema-constrained sampling across vLLM API generations."""
    from vllm import SamplingParams
    last: Any = None
    try:  # vLLM >= 0.11
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              structured_outputs=StructuredOutputsParams(json=LLM_JSON_SCHEMA))
    except Exception as exc:  # noqa: BLE001
        last = exc
    try:  # vLLM 0.6 - 0.10
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_decoding=GuidedDecodingParams(json=LLM_JSON_SCHEMA))
    except Exception as exc:  # noqa: BLE001
        last = exc
    try:  # vLLM < 0.6
        return SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              guided_json=LLM_JSON_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        last = exc
    msg = f"no JSON-schema decoding API in this vLLM build (last error: {last})"
    if require_schema:
        raise RuntimeError(msg + " -- pass --allow-unconstrained to run anyway")
    print(f"[WARN] {msg}; falling back to regex parsing")
    return SamplingParams(temperature=temperature, max_tokens=max_tokens)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified CBS + Eurostat enrichment (vLLM)")
    ap.add_argument("--source", choices=["cbs", "eurostat", "both"], default="both")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output", default=None, type=Path,
                    help="default: data/processed/enriched_unified_<model>.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="max items PER SOURCE")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=int(os.environ.get("TP_SIZE", "1")))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=250,
                    help="Flush results to disk every N datasets (interruption safety)")
    ap.add_argument("--allow-unconstrained", action="store_true")
    ap.add_argument("--include-ungrounded", action="store_true",
                    help="Eurostat: also enrich datasets that have no harvested structure metadata")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build prompts and report stats; no GPU, no generation")
    args = ap.parse_args()

    out = args.output or Path("data/processed") / (
        "enriched_unified_" + args.model.split("/")[-1].replace(".", "").lower() + ".jsonl")
    items = collect_items(args.source, args.limit,
                          require_metadata=not args.include_ungrounded)
    done = already_done(out) if args.resume else set()
    items = [it for it in items if (it["deterministic"]["publisher"], it["code"]) not in done]

    prompts = [USER_TEMPLATE.format(code=it["code"], **it["prompt_fields"]) for it in items]
    n_grounded = sum(1 for it in items if it["deterministic"]["grounded"])
    by_pub: Dict[str, int] = {}
    for it in items:
        by_pub[it["deterministic"]["publisher"]] = by_pub.get(it["deterministic"]["publisher"], 0) + 1
    print(f"[INFO] {len(items)} datasets to enrich {by_pub} | grounded in real data: "
          f"{n_grounded} ({n_grounded / max(len(items), 1):.0%}) | skipped {len(done)} done")
    print(f"[INFO] model={args.model} -> {out}")
    if args.dry_run:
        validate_schema_both_backends(args.model)
        lens = [len(p) for p in prompts]
        if lens:
            print(f"[DRY-RUN] prompt chars: median {sorted(lens)[len(lens)//2]}, max {max(lens)}")
            print("\n--- sample prompt ---\n" + prompts[0][:1200])
        return
    if not items:
        print("[DONE] nothing to do")
        return

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tensor_parallel_size, trust_remote_code=True, **llm_kwargs())
    sampling = make_sampling_params(args.temperature, args.max_tokens,
                                    require_schema=not args.allow_unconstrained)
    conversations = [[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": p}] for p in prompts]

    def generate(convs):
        try:
            # Qwen3 is a hybrid reasoning model: without this it emits a <think>
            # block, which wastes the token budget and fights the JSON constraint.
            return llm.chat(convs, sampling, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            return llm.chat(convs, sampling)

    out.parent.mkdir(parents=True, exist_ok=True)
    ok, failures = 0, []
    t0 = time.time()
    # Generate in chunks and flush after each: a single batch over ~1.9k prompts
    # writes nothing until the very end, so any interruption loses the whole run.
    # Chunked, a killed or preempted job keeps everything already written and
    # --resume picks up from there.
    mode = "a" if (args.resume and out.exists()) else "w"
    with out.open(mode, encoding="utf-8") as fout:
        for start in range(0, len(items), args.chunk_size):
            chunk = items[start:start + args.chunk_size]
            outputs = generate(conversations[start:start + args.chunk_size])
            for it, o in zip(chunk, outputs):
                text = o.outputs[0].text if o.outputs else ""
                obj = parse_json(text)
                if not obj or (LLM_REQUIRED_KEYS - set(obj.keys())):
                    failures.append({"code": it["code"],
                                     "publisher": it["deterministic"]["publisher"],
                                     "reason": "unparseable" if not obj else
                                               f"missing: {sorted(LLM_REQUIRED_KEYS - set(obj.keys()))}",
                                     "raw": text[:500]})
                    continue
                rec = finalize_record(obj, it["deterministic"], args.model)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                ok += 1
            fout.flush()
            os.fsync(fout.fileno())
            done_n = min(start + args.chunk_size, len(items))
            el = time.time() - t0
            print(f"[{done_n}/{len(items)}] ok={ok} fail={len(failures)} | "
                  f"{el/60:.1f}m elapsed, {done_n/max(el,1):.2f} datasets/s", flush=True)
    dt = time.time() - t0
    print(f"[DONE] enriched ok={ok} fail={len(failures)} in {dt/60:.1f}m -> {out}")
    if failures:
        fp = out.with_suffix(out.suffix + ".failures.json")
        fp.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[WARN] {len(failures)} failed -> {fp}; --resume retries exactly these")


if __name__ == "__main__":
    main()
