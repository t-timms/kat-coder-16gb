"""THE moment of truth: does the pruned+quantized KAT-Coder serve on SM120?

13.28 GiB against 16.3 GB of VRAM, so this loads ENTIRELY on the GPU. That removes
the CPU-offload UVA path that crashed the 21.9 GB control checkpoint
(uva.py:119, illegal memory access) and finally exercises the NVFP4 kernels
themselves on qwen3_5_moe.

If this produces coherent code, the product path is proven end to end:
  REAP 50% prune -> NVFP4A16 -> vLLM on a consumer Blackwell card, in 16 GB.

Non-negotiables:
  * enforce_eager=True - CUDA graph capture is numerically broken on SM120.
  * language_model_only=True - otherwise vLLM profiles a 16384-token image budget
    through a vision tower that has ZERO trained weights, and grinds for 16+ min.
  * llm.chat(), never apply_chat_template()+generate() - double-BOS fakes corruption.
  * Judge the CONTENT. ZAYA1's failure mode was all-pad output that still "succeeded".
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

MODEL = os.path.expanduser("~/models/kat-50pct-nvfp4a16")

PROMPTS = [
    "Write a Python function that reverses a singly linked list. Return only code.",
    "Write a Python function `is_palindrome(s)` ignoring case and non-alphanumerics.",
    "Write a Python function that merges two sorted lists into one sorted list.",
    "Fix the bug:\n\ndef avg(xs):\n    return sum(xs) / len(xs)\n\nExplain briefly, then give the fixed code.",
]


def main() -> None:
    from vllm import LLM, SamplingParams

    print(f"model: {MODEL}", flush=True)
    t0 = time.time()

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        enforce_eager=True,
        # NO cpu_offload_gb - 13.28 GiB fits natively. This is the whole point.
        # 0.95 asked for 15.12 GiB but only 14.66 of 15.92 GiB is free - the Windows
        # desktop holds ~1.26 GiB. 0.90 gives a 14.33 GiB budget against 13.28 GiB of
        # weights, leaving ~1 GiB, which is plenty: KV is ~10 KB/token across the 10
        # of 40 layers that hold it, so 2048 tokens costs ~20 MB.
        gpu_memory_utilization=0.90,
        language_model_only=True,
        trust_remote_code=True,
    )
    print(f"LOAD_OK in {time.time() - t0:.1f}s", flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=300)
    convs = [[{"role": "user", "content": p}] for p in PROMPTS]

    t1 = time.time()
    outs = llm.chat(convs, params)
    dt = time.time() - t1
    total_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"GENERATE_OK in {dt:.1f}s  ({total_tok} tokens, {total_tok / dt:.1f} tok/s aggregate)", flush=True)

    healthy = 0
    for i, out in enumerate(outs):
        text = out.outputs[0].text
        ids = out.outputs[0].token_ids
        print("\n" + "=" * 70)
        print(f"PROMPT {i}: {PROMPTS[i][:70]}")
        print(f"finish_reason: {out.outputs[0].finish_reason}  tokens: {len(ids)}")
        print("-" * 70)
        print(text[:1000])

        distinct = len(set(ids))
        has_code = ("def " in text) or ("return" in text)
        if distinct <= 3:
            print(f"  !! COLLAPSED: only {distinct} distinct token ids")
        elif not has_code:
            print("  !! no recognisable code")
        else:
            print(f"  OK: {distinct} distinct tokens, contains code")
            healthy += 1

    print("\n" + "=" * 70)
    print(f"VERDICT: {healthy}/{len(PROMPTS)} healthy")
    print("SMOKE_PASS" if healthy == len(PROMPTS) else "SMOKE_FAIL", flush=True)
    if healthy != len(PROMPTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
