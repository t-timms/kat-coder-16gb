# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **SWE-bench Verified 50-instance results:** 20/50 = 40.0% under the standard
  metric, 20/22 = 90.9% when the model produces a valid patch. Bottlenecks:
  18 ContextWindowExceeded (32K ceiling), 9 LimitsExceeded, 1 garbage patch
  out of 23 generated. Results in
  `results/hosted_vllm__kat-16gb.kat-coder-16gb-50.json`.
- **`kat_overrides_context_managed.yaml`, an opt-in, unvalidated alternative
  agent config** (`step_limit=30`, `max_tokens=1024`, 5K observation
  truncation, down from 40/3072/10K) aimed at the 18 ContextWindowExceeded
  failures above. Select it with `KAT_CONFIG=kat_overrides_context_managed.yaml`
  when invoking `run_pilot_all.sh`; default behavior is unchanged. Tightening
  `max_tokens` this far risks truncating the model's mandatory `<think>` trace
  mid-turn, which would show up as format failures instead of CWEs rather than
  as a net win — this needs a real run before it's trusted either way.

- **Model card published to Hugging Face:** [`Ttimms/kat-coder-16gb`](https://huggingface.co/Ttimms/kat-coder-16gb),
  content identical to `HF_MODEL_CARD.md` in this repo. Checkpoint upload
  (`kat-50pct-nvfp4a16-renorm-stripped`) is pending — it has to be pushed
  from the machine holding the weights.

### Changed

- **`max_num_seqs` 2 → 8** in `serve_kat.sh` and `run_pilot_all.sh`, tested
  2026-08-19: 1.86x concurrency headroom at the 32K context length with no
  reduction in per-sequence KV budget. Throughput/concurrency only — does not
  change the SWE-bench score above.

## [0.2.0] - 2026-08-18

Accuracy measured, agentic serving solved, SWE-bench pipeline wired end to end.
The model is now a measured agentic coder, not just a fast one.

### Added

- **Accuracy results.** HumanEval+ 89.0% [83.3, 92.9] and MBPP+ 90.5% [87.1, 93.0]
  on the release candidate, greedy decoding, instruct framing. ~8 min for 706
  problems, only affordable with CUDA graphs enabled.
- **Agentic serving config.** PIECEWISE graph mode costs 7% speed and buys 4.4x
  context (64,976 vs 14,672 tokens). Prefix caching is worth 45x on replayed
  history (0.21 s vs 30.74 s for a 13K-token history). Required flags:
  `--enable-prefix-caching --max-num-batched-tokens 4096` (the 2048 default sits
  48 tokens under the Mamba block_size assertion).
- **SWE-bench Verified pipeline** via mini-swe-agent 2.4.6 + swebench 5.0.1.
  Scripts: `run_pilot_all.sh` (serve + rollout + teardown), `grade_pilot.sh`
  (official harness), `preflight_litellm.py` (validates through litellm, not curl).
- **Pilot results:** 5 instances, 4 Submitted, 1 ContextWindowExceeded, 4/4
  patches resolved by the official grading harness.
- `eval_suite.sh`, `read_scores.py`, `inspect_gen.py` for running EvalPlus
  benchmarks (HumanEval+, MBPP+) through lm-eval-harness.
- `analyze_pilot.py` for analyzing SWE-bench rollout trajectories (exit statuses,
  step counts, context growth per instance).

### Changed

- **Speed re-measured on the actual release candidate:** 149.5 tok/s (n=5, range
  [1.691, 1.777] s). The earlier 146.4 tok/s figure was measured on the pre-renorm
  pre-strip checkpoint and transfers, but had never been measured on the artifact
  we would ship.
- README rewritten to reflect current status: results table, agentic serving docs,
  SWE-bench pipeline, corrected competitive bar.
- `serve_kat.sh` now includes `--enable-auto-tool-choice --tool-call-parser
  qwen3_xml` (required by mini-swe-agent's litellm model class, which sends
  `tools=[BASH_TOOL]` unconditionally regardless of prompt config).

### Fixed

- **SWE-bench tool calling was silently broken.** mini-swe-agent's default model
  class (`models/litellm_model.py:69`) sends `tool_choice: auto` unconditionally.
  Without `--enable-auto-tool-choice` on the server, every LM call returned 400.
  A curl-based preflight passed while the litellm path was broken, because curl
  skipped the layer with the bug. New preflight (`preflight_litellm.py`) goes
  through litellm with the same BASH_TOOL definition the agent uses.
- SWE-bench grading used wrong dataset name (`princeton-nlp/SWE-bench_Verified`
  instead of `SWE-bench/SWE-bench_Verified` for swebench 5.0.1).
- `--cache_level` flag removed from grade_pilot.sh (does not exist in swebench
  5.0.1, left over from older docs).

### Known limitations

- **32K context window** is the safe ceiling (KV budget fluctuates 0.49-1.41 GiB
  with the Windows desktop's VRAM). Devstral Small averages 86.9 LM calls/instance
  and many instances will hit the context limit before completing. The run must be
  disclosed as step-limited.
- **No pruning baseline.** The unpruned model is 69.3 GB bf16 and cannot fit this
  machine. Absolute scores (89.0% / 90.5%) are measured but the pruning cost is
  not. A cloud run for the baseline arm is the cheapest path (~$4-6).
- SWE-bench Verified no longer accepts leaderboard submissions outside academia.
- `KAT-Coder-V2.5-Dev` publishes no HumanEval/MBPP/EvalPlus, so there is no
  published number to compare our 89.0% / 90.5% against.

## [0.1.0] - 2026-08-17

First working pipeline. `Kwaipilot/KAT-Coder-V2.5-Dev` runs inside 16 GB of
consumer VRAM and produces correct code.

### Added

- REAP expert pruning at 50 percent for `qwen3_5_moe`, which required adding
  Qwen3.5/3.6 MoE support to the reap fork. llm-compressor's own REAP modifier
  rejects this architecture: it detects MoE layers by duck typing and requires
  `LinearExperts2D`, which Qwen3.5's fused `Qwen3_5MoeExperts` is not, so all 40
  layers are skipped.
- NVFP4 quantization in two schemes via llm-compressor. `NVFP4A16` is weight-only
  and data free, completing in 82 seconds. `NVFP4` is W4A4, requires calibration,
  and takes 28.7 minutes. Both produce the same size, 13.28 versus 13.29 GiB,
  because activations are never stored.
- Paired evaluation harness: held-out code perplexity via a custom
  `loglikelihood_rolling` lm-eval task, Wilcoxon and McNemar tests, bootstrap
  confidence intervals, and a resolution diagnostic that reports when a null
  result comes from an underpowered test rather than a real absence of effect.
- A/B latency benchmarking built on `vllm bench`, with repetition across separate
  process invocations, interleaved arms, and a discarded warmup.
- Precondition probes that run before expensive jobs, covering dataset
  availability, quantization toolchain readiness, and architecture support.

### Fixed

- Router renormalization was silently disabled during saliency computation. reap
  gated it on `getattr(config, "norm_topk_prob", False)`, but Qwen3.5 renormalizes
  unconditionally inside `Qwen3_5MoeTopKRouter.forward` and omits the flag from its
  config. Output directories were still named from the requested value, so runs
  appeared correctly configured while renormalization was off. Now resolved by
  asking the adapter, which knows what the architecture does. Committed separately
  for upstreaming.
- Pruned checkpoints were missing four files the loader requires:
  `preprocessor_config.json`, `video_preprocessor_config.json`, `merges.txt` and
  `vocab.json`. reap's save path drops them.
- Scripts hardcoded a home directory, making a clone unrunnable by anyone else.

### Verified on RTX 5070 Ti (SM120)

- 50 percent pruned plus NVFP4 is **13.28 GiB**, loads in **31 s** with no CPU
  offload, and generates correct code. NVFP4 compute is numerically correct on
  SM120 for `qwen3_5_moe`.
- Serving requires `enforce_eager=True`, `language_model_only=True`, and
  `gpu_memory_utilization=0.90`. `cpu_offload_gb` crashes in the UVA path and is
  unnecessary once pruned.
- The model declares a vision tower it has no trained weights for: 31,333 tensors,
  all under `model.language_model.`, none matching visual or vision.

### Known limitations

- Accuracy is unmeasured. No HumanEval or agentic benchmark has been run.
- Throughput is roughly 19 tok/s single stream, well short of the reported
  envelope for this architecture on comparable hardware.
- The current checkpoints were pruned before the renormalization fix, so they are
  a proof of path rather than a release build.
