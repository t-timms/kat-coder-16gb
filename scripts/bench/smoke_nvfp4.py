"""Does a compressed-tensors NVFP4 MoE checkpoint actually SERVE on SM120?

This is the control arm the KAT-Coder plan designated and never ran. It uses
sakamakismile/KAT-Coder-V2.5-Dev-NVFP4, already on disk, built with llm-compressor
(QuantizationModifier scheme=NVFP4) - the exact toolchain we intend to use. So it
tests our pipeline's output format without us building anything.

It answers, in one run:
  1. Does vLLM load compressed-tensors NVFP4 MoE for qwen3_5_moe on SM120, or hit
     the known fused-MoE initialization limitation?
  2. Does it emit coherent CODE, or the pad-token collapse that plagued ZAYA1?
  3. Is W4A4 viable for coding on this model? (this checkpoint is W4A4:
     group_0 weights 4 / acts 4)

Non-negotiables on this machine:
  * enforce_eager=True - CUDA graph capture is numerically broken on SM120 for
    every MoE and attention backend. Without it, output corruption is guaranteed
    and would be misread as a bad checkpoint.
  * llm.chat(), never apply_chat_template() + generate() - the double-BOS bug
    fakes corruption convincingly.
  * 21.9 GB against 16.3 GB VRAM, so CPU offload is required. Slow is fine; we
    are testing a code path, not measuring throughput.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "0")

MODEL = os.path.expanduser("~/models/KAT-Coder-V2.5-Dev-NVFP4")

PROMPTS = [
    "Write a Python function that reverses a linked list. Return only the code.",
    "Write a Python function `is_palindrome(s)` that ignores case and non-alphanumeric characters.",
    "In Python, write a function that merges two sorted lists into one sorted list.",
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
        cpu_offload_gb=16,
        # 0.85 left only 345 MiB free on a 16.3 GB card, and warmup then ground for
        # 16+ minutes without finishing. Weights stream over PCIe via UVA, so the
        # allocator needs real headroom to work in.
        gpu_memory_utilization=0.75,
        # The first attempt stalled in "profiled with 1 image items of the maximum
        # feature size" with a 16384-token encoder budget. KAT-Coder has ZERO trained
        # vision weights (31,333 keys, all model.language_model.*) - the tower exists
        # only in the config shell, so that profiling was grinding on random noise.
        language_model_only=True,
        trust_remote_code=True,
    )
    print(f"LOAD_OK in {time.time() - t0:.1f}s", flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=256)
    convs = [[{"role": "user", "content": p}] for p in PROMPTS]

    t1 = time.time()
    outs = llm.chat(convs, params)
    print(f"GENERATE_OK in {time.time() - t1:.1f}s", flush=True)

    healthy = 0
    for i, out in enumerate(outs):
        text = out.outputs[0].text
        ids = out.outputs[0].token_ids
        print("\n" + "=" * 70)
        print(f"PROMPT {i}: {PROMPTS[i]}")
        print(f"finish_reason: {out.outputs[0].finish_reason}  tokens: {len(ids)}")
        print("-" * 70)
        print(text[:900])

        # Health checks. ZAYA1's failure mode was all-pad output that still
        # "succeeded", so judge the CONTENT, not the return code.
        distinct = len(set(ids))
        has_code = ("def " in text) or ("return" in text)
        collapsed = distinct <= 3
        if collapsed:
            print(f"  !! COLLAPSED: only {distinct} distinct token ids")
        elif not has_code:
            print("  !! no recognisable code in output")
        else:
            print(f"  OK: {distinct} distinct tokens, contains code")
            healthy += 1

    print("\n" + "=" * 70)
    print(f"VERDICT: {healthy}/{len(PROMPTS)} prompts produced healthy code")
    if healthy == len(PROMPTS):
        print("SMOKE_PASS")
    else:
        print("SMOKE_FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
