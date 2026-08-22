---
base_model:
  - Kwaipilot/KAT-Coder-V2.5-Dev
base_model_relation: quantized
license: apache-2.0
library_name: transformers
datasets:
  - theblackcat102/evol-codealpaca-v1
tags:
  - reap
  - pruning
  - nvfp4
  - nvfp4a16
  - fp4
  - 4-bit
  - compressed-tensors
  - quantization
  - vllm
  - blackwell
  - moe
  - agentic-coding
  - swe-bench
pipeline_tag: text-generation
---

# KAT-Coder-V2.5-Dev REAP-50 NVFP4A16 (16 GB)

**REAP expert-pruned (50%) + NVFP4A16 quantized build of `Kwaipilot/KAT-Coder-V2.5-Dev`
(69.40 SWE-bench Verified claimed), sized and served to run as a local agentic
coding model inside 16 GB of consumer VRAM** — 12.45 GiB, RTX 5070 Ti (SM120),
vLLM. Built with a router-renormalization fix for this architecture (contributed
upstream) and a vision tower stripped of its untrained weights.

**SWE-bench Verified: 26/50 = 52.0% resolved**, via `mini-swe-agent`'s official
bash-only scaffold, at a 49K-token context ceiling — up from 40.0% at the
original 32K ceiling tested on this same checkpoint. Still below the 56.4% bar
set by Devstral Small (2512) under the same scaffold, but closed most of the
gap. 17 of 50 instances produced no usable patch, all from hitting the context
ceiling; the run must still be read as context-limited, not as an
unconditional capability measurement. See "SWE-bench Verified" below before
citing the headline number without that context.

## Highlights

| Result | Detail |
|---|---|
| **12.45 GiB** | REAP 50% expert pruning + NVFP4A16 (weight-only, data-free), vision tower stripped |
| **149.5 tok/s** median, n=5 | Benchmark config: 512 in / 256 out, batch 1, CUDA graphs (FULL_AND_PIECEWISE), 14,672-token context ceiling |
| **89.0% / 90.5%** | HumanEval+ [83.3, 92.9] / MBPP+ [87.1, 93.0], greedy, instruct framing |
| **52.0%** (26/50) | SWE-bench Verified, `mini-swe-agent` bash-only, 49K context — see caveats below |
| **28.9 s load** | CUDA graphs enabled, no CPU offload |

## Why 50 percent

Forced by arithmetic on a 16 GB card, not a tuning choice:

| variant | size | fits 16 GB |
|---|---:|:---:|
| bf16 base | 69.3 GB | no |
| NVFP4, unpruned | 21.9 GB | no |
| REAP 25% + NVFP4 | ~16–17 GB | no — not once KV cache is counted |
| **REAP 50% + NVFP4** | **12.45 GiB** | **yes** |

Supporting evidence: [Half the Experts, All the Code](https://arxiv.org/html/2607.16721)
pruned Qwen3.6-35B-A3B, this model's own base, at 50% with no statistically
detectable loss on its primary code benchmark.

## SWE-bench Verified — read before citing the 52.0% figure alone

| metric | value |
|---|---:|
| resolved | **26/50 = 52.0%** |
| resolved of completed (valid patch produced and tested) | 26/32 = **81.25%** |
| ContextWindowExceeded | 17 (49K ceiling) |
| garbage/invalid patch (patch failed to apply) | 1 of 33 generated |
| ran, tests failed (genuinely unresolved) | 6 |

Prior measurement at this checkpoint's original 32K-token ceiling: 20/50 =
40.0% (18 ContextWindowExceeded, 20/22 = 90.9% resolved-of-completed). Raising
the ceiling to 49K (`max_model_len 49152`, `max_num_seqs 2`) moved the
headline number from 40.0% to 52.0%, but **not** primarily by reducing
context-window failures — the failure rate barely moved (17/50 = 34% vs.
18/50 = 36%). The gain came from instances that fit within the ceiling either
way resolving far more often with more room to work in: 81.25%
resolved-of-completed at 49K vs. 62.5% at 32K. The scaffold's context ceiling
is still the safe limit this card's VRAM budget supports, not a property of
the model — Devstral Small averages 86.9 LM calls/instance under the same
scaffold, and instances needing more than 49K tokens of context still fail to
submit at all. This is disclosed as a real result, not an excuse: 52.0% is the
correct number to cite; the breakdown above is the correct context for
interpreting it.

## Prior art and scope of claims

Verified against the Hugging Face Hub on 2026-08-17:

- REAP combined with NVFP4 on `qwen3_5_moe` already exists
  ([`rene98c/Qwen3.5-397B-A17B-REAP-28-NVFP4`](https://huggingface.co/rene98c/Qwen3.5-397B-A17B-REAP-28-NVFP4),
  March 2026, 23.1K downloads).
- REAP on this specific model exists as GGUF
  ([`gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF`](https://huggingface.co/gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF)).

What is distinct, and all that is claimed: a **vLLM-servable KAT-Coder that is
genuinely usable in 16 GB**, with published SWE-bench Verified, HumanEval+, and
MBPP+ numbers and their confidence intervals — none of which the prior art
above publishes.

## Quantization and pruning details

| Field | Value |
|---|---|
| Base model | `Kwaipilot/KAT-Coder-V2.5-Dev` |
| Pruning | REAP, expert-level, 50% compression ratio, seed 42 |
| Pruning calibration | `theblackcat102/evol-codealpaca-v1`, 64 batches/category, 2048 max length |
| Router renormalization | Fixed (was silently disabled by the upstream REAP adapter for this architecture; committed for upstreaming) |
| Quantization method | compressed-tensors / llm-compressor, `QuantizationModifier` (PTQ) |
| Quantization scheme | NVFP4A16 — weight-only, data-free, 82 s |
| Quantization calibration | `evol-codealpaca` (deliberately not the Magicoder set used for evaluation) |
| Ignored / kept unquantized | `lm_head`, routers, shared expert gates, embeddings, DeltaNet conv1d + linear-attention projections, MTP module |
| Vision tower | Removed. The base model declares one and ships no weights for it; this checkpoint contains neither the declaration nor the 333 untrained tensors (0.83 GiB) transformers would otherwise materialise. |
| Files | 3 safetensors shards of ~5 GiB plus `model.safetensors.index.json`, matching the base model's layout, so an interrupted download resumes at shard granularity |
| Built on | RTX 5070 Ti, 16 GB VRAM, SM120 (compute capability 12.0) |
| Serving kernel | vLLM selects `MarlinNvFp4LinearKernel` on this card: 4-bit weights are decoded and the GEMM runs in bf16. NVFP4A16 is weight-only, so no FP4 arithmetic is required. A native FP4 path for SM120 exists via FlashInfer CuTeDSL but is not used here and is unmeasured. |

## Usage

Requires vLLM with SM120 support (CUDA graphs are correct on this card for
this model, despite past reports of SM120 CUDA-graph issues on other
architectures) and native tool calling for agentic use:

```bash
vllm serve Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16 \
  --served-model-name kat-16gb \
  --max-model-len 32768 --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --enable-prefix-caching --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
  --language-model-only
```

`--language-model-only` is required: the model declares a vision tower it has
no trained weights for, and without this flag vLLM profiles a 16K-token image
budget through it. `--enable-prefix-caching` is the single largest agentic
lever measured on this model — 45x on replayed history (0.21 s vs 30.74 s for
a 13,130-token history). Never set `--max-model-len` near a measured ceiling:
available KV cache swings 0.49–1.41 GiB with host desktop VRAM use, and higher
values fail intermittently rather than at startup.

## Evaluation

HumanEval+ and MBPP+ via lm-eval-harness / EvalPlus, greedy decoding, instruct
framing, Wilson confidence intervals:

| benchmark | score | 95% CI | n |
|---|---:|---:|---:|
| HumanEval+ | 89.0% | [83.3, 92.9] | 164 |
| MBPP+ | 90.5% | [87.1, 93.0] | 378 |

`KAT-Coder-V2.5-Dev` publishes no HumanEval/MBPP/EvalPlus numbers, so there is
no published upstream figure to compare these against.

Both figures were re-measured on the released checkpoint itself and reproduced
inside their intervals: HumanEval+ 90.9% [85.5, 94.4] and MBPP+ 89.9%
[86.5, 92.6], same problem counts, same greedy decoding. The table reports the
original measurement. The two differences run in opposite directions
(+1.9 pp and -0.6 pp), which is greedy-decoding nondeterminism under vLLM's
batching rather than a difference in weights. Reproduce with
`bash scripts/eval/eval_suite.sh`.

SWE-bench Verified via the official `swebench.harness.run_evaluation` harness
against `mini-swe-agent` bash-only rollouts (scaffold: SWE-bench/experiments
v1.17.2 configuration) — see the dedicated section above for the full
breakdown and required caveats.

## Known limitations

- **49K context window** (raised from an earlier 32K ceiling) is a
  hardware-forced limit, not a design choice — this card's KV-cache budget
  cannot safely support more. SWE-bench results must still be read as
  context-limited: 17 of 50 instances in the reported run failed purely from
  exceeding this ceiling, not from the model failing the task.
- **No pruning-ablation baseline measured.** The unpruned model is 69.3 GB
  bf16 and does not fit this hardware; the accuracy cost of pruning itself
  (independent of quantization) is not isolated here.
- SWE-bench Verified no longer accepts leaderboard submissions outside
  academia — these numbers are self-reported and independently reproducible
  from the released evaluation scripts, not a leaderboard entry.

## W4A4: an alternative quantization strategy

A follow-up build, [`Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4),
re-quantizes this same pruned checkpoint to `NVFP4` (W4A4 — weights **and**
activations to 4 bits) to reach native FP4 tensor-core kernels instead of
this model's Marlin dequant-to-bf16 fallback. Measured properly (5
interleaved runs per arm, median + range, under both isolated eager mode
and the same PIECEWISE CUDA-graph configuration this model actually serves
with): W4A4 decodes at 119.2 tok/s vs this checkpoint's 142.5 tok/s (0.84x)
with a mixed accuracy picture (HumanEval 92.07% vs 95.7%, HumanEval+ 89.02%
vs 90.9%, MBPP+ 91.01% vs 89.9% — that last one favors W4A4). This
checkpoint is faster on this hardware and is what we default to; the W4A4
build is a legitimate alternative if the native FP4 execution path matters
more for your use case. Only single-stream (batch=1) decode was measured.
Full writeup:
[`ROADMAP.md`](https://github.com/t-timms/moe-pruning-nvfp4/blob/main/ROADMAP.md)'s
RESULT section in the release repo.

## License

Apache 2.0, inherited from the base model `Kwaipilot/KAT-Coder-V2.5-Dev` and
matching the `reap` and `llm-compressor` toolchains used to build this checkpoint.

## Citation

This checkpoint is derived from `Kwaipilot/KAT-Coder-V2.5-Dev`. If you use it,
please cite the upstream technical report:

```bibtex
@misc{katcoder_v25_2026,
  title={{KAT-Coder-V2.5 Technical Report}},
  author={{KwaiKAT Team}},
  year={2026},
  month={July},
  eprint={2607.05471},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/pdf/2607.05471}
}
```
