# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Quickstart and reproduce instructions in the README.
- `docs/environment.md` describing the three required Python environments, why
  they cannot be merged, and the WSL-specific traps.
- Apache 2.0 licence, matching `reap` and `llm-compressor` so the router
  renormalization fix can be offered upstream without a licence mismatch.

### Changed
- Shell scripts are now executable.

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
  all under `model.language_model.`, none matching visual or vision. The declared
  tower is randomly initialised on every load and caused three separate failures.

### Known limitations
- Accuracy is unmeasured. No HumanEval or agentic benchmark has been run.
- Throughput is roughly 19 tok/s single stream, well short of the reported
  envelope for this architecture on comparable hardware. The gap appears to be in
  the serving stack rather than the quantization scheme.
- The current checkpoints were pruned before the renormalization fix, so they are
  a proof of path rather than a release build.
