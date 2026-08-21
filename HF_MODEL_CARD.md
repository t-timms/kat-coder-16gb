---
base_model:
  - Kwaipilot/KAT-Coder-V2.5-Dev
base_model_relation: quantized
license: apache-2.0
tags:
  - reap
  - pruning
  - nvfp4
  - nvfp4a16
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

**SWE-bench Verified: 20/50 = 40.0% resolved**, via `mini-swe-agent`'s official
bash-only scaffold — below the 56.4% bar set by Devstral Small (2512) under the
same scaffold. 27 of 50 instances produced no usable patch; 18 of those hit the
32K context ceiling this card's VRAM budget imposes, and the run must be read
as context/step-limited, not as an unconditional capability measurement. See
"SWE-bench Verified" below before citing the headline number without that
context.

## Highlights

| Result | Detail |
|---|---|
| **12.45 GiB** | REAP 50% expert pruning + NVFP4A16 (weight-only, data-free), vision tower stripped |
| **149.5 tok/s** median, n=5 | Benchmark config: 512 in / 256 out, batch 1, CUDA graphs (FULL_AND_PIECEWISE), 14,672-token context ceiling |
| **89.0% / 90.5%** | HumanEval+ [83.3, 92.9] / MBPP+ [87.1, 93.0], greedy, instruct framing |
| **40.0%** (20/50) | SWE-bench Verified, `mini-swe-agent` bash-only — see caveats below |
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

## SWE-bench Verified — read before citing the 40.0% figure alone

| metric | value |
|---|---:|
| resolved | **20/50 = 40.0%** |
| resolved of completed (valid patch produced) | 20/22 = **90.9%** |
| ContextWindowExceeded | 18 (32K ceiling) |
| LimitsExceeded | 9 |
| garbage/invalid patch | 1 of 23 generated |

The scaffold is capped at a 32K-token context window — the safe ceiling this
card's VRAM budget supports, not a property of the model. Devstral Small
averages 86.9 LM calls/instance under the same scaffold; many KAT-Coder
instances hit the context limit before finishing. **90.9% of instances where
the model actually produced a patch had that patch resolve the issue** — most
of the gap to the 56.4% competitive bar is instances that never got to submit
a patch at all, not patches that were wrong. This is disclosed as a real
result, not an excuse: the 40.0% headline number is the correct number to
cite; the breakdown above is the correct context for interpreting it.

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
| Vision tower | Removed — both the config declaration and the 333 untrained tensors (0.83 GiB) that transformers materialises for a tower the base model ships no weights for. Stripping only the config leaves the weights in the file; this checkpoint has neither. |
| Built on | RTX 5070 Ti, 16 GB VRAM, SM120 |

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

SWE-bench Verified via the official `swebench.harness.run_evaluation` harness
against `mini-swe-agent` bash-only rollouts (scaffold: SWE-bench/experiments
v1.17.2 configuration) — see the dedicated section above for the full
breakdown and required caveats.

## Known limitations

- **32K context window** is a hardware-forced ceiling, not a design choice —
  this card's KV-cache budget cannot safely support more. SWE-bench results
  must be read as context/step-limited.
- **No pruning-ablation baseline measured.** The unpruned model is 69.3 GB
  bf16 and does not fit this hardware; the accuracy cost of pruning itself
  (independent of quantization) is not isolated here.
- SWE-bench Verified no longer accepts leaderboard submissions outside
  academia — these numbers are self-reported and independently reproducible
  from the released evaluation scripts, not a leaderboard entry.

## License

Apache 2.0, matching the `reap` and `llm-compressor` toolchains used to build
this checkpoint.

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
