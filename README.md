# kat-coder-16gb

Making `Kwaipilot/KAT-Coder-V2.5-Dev` (69.40 SWE-bench Verified) run as a usable
local agentic coding model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120).

Pipeline: REAP expert pruning at 50 percent, then NVFP4 quantization, served by vLLM.

## Status

Working proof of path, not a release.

| stage | status |
|---|---|
| REAP 50% prune (Qwen3.5 MoE support added to reap fork) | done, both seeds |
| NVFP4 quantization | done, two variants built |
| Loads and generates correct code in 16 GB | **yes, verified** |
| Router renormalization fix | done, verified |
| Speed benchmarked properly | in progress |
| HumanEval / agentic accuracy | not started |
| Release checkpoint | not built |

The 50 percent pruned model quantizes to **13.28 GiB**, loads in **31 s** with no
CPU offload, and produces correct code (verified by reading it: a three-pointer
linked-list reversal, a palindrome checker with type hints, and a correct
`ZeroDivisionError` diagnosis with guard).

**NVFP4 compute is numerically correct on SM120 for `qwen3_5_moe`.** No pad
collapse, no NaN. That was the project's central risk and it did not materialise.

## Quickstart

Environment setup, including why three separate Python environments are required,
is in [docs/environment.md](docs/environment.md). Expect to need an RTX 5070 Ti or
another SM120 card, about 250 GB of disk, and 80 GB allocated to WSL2.

**1. Check preconditions before spending hours on a run**

```bash
bash scripts/probes/check_quant_preconditions.sh
```

Reports which environment has llm-compressor, whether the calibration artifacts
exist, disk headroom, and free VRAM, in a single pass. Discovering these one
failure at a time costs a load cycle each.

**2. Prune to 50 percent**

```bash
bash scripts/prune/prune_and_eval_50.sh
bash scripts/prune/fix_ckpt_files.sh     # restore files reap's save path drops
```

Roughly 3 minutes given cached calibration observations, or about an hour if
calibration has to run. 50 percent is not a tuning choice: unpruned NVFP4 is
21.9 GB and 25 percent lands at 16-17 GB, neither of which fits once KV cache is
counted.

**3. Quantize**

```bash
~/quant-env/bin/python scripts/quantize/quantize_kat.py        # NVFP4A16, 82 s
~/quant-env/bin/python scripts/quantize/quantize_kat_w4a4.py   # W4A4, 28.7 min
```

Weight-only is data free and therefore fast. W4A4 quantizes activations and needs
real calibration. Both output about 13.3 GiB.

**4. Confirm it serves and produces real code**

```bash
~/vllm-env/bin/python scripts/bench/smoke_pruned_nvfp4.py
```

Judges the output by content, not exit code, because this stack returns 0 on
failure often enough that exit codes are not evidence.

**5. Benchmark**

```bash
bash scripts/bench/bench_ab.sh 5
~/vllm-env/bin/python scripts/bench/bench_ab_analyze.py
```

Reports median and range over separate process invocations, and prints which
kernel each arm actually selected. That last part matters: a silent fallback to an
unsupported kernel appears only as lost throughput, with no error.

**6. Compare two checkpoints properly**

```bash
bash scripts/eval/paired_eval.sh 1000
~/reap-env/bin/python scripts/eval/paired_analyze.py
```

Pairs by document hash and verifies both models scored identical text rather than
assuming it, then reports a resolution diagnostic alongside the p-value so an
underpowered null is not mistaken for evidence of no difference.

## Why 50 percent

Forced by arithmetic, not chosen:

| variant | size | fits 16 GB |
|---|---:|---|
| bf16 base | 69.3 GB | no |
| NVFP4, unpruned | 21.9 GB | no |
| REAP 25% + NVFP4 | ~16-17 GB | no, not once KV cache is counted |
| **REAP 50% + NVFP4** | **13.3 GiB** | **yes** |

Supporting evidence: [Half the Experts, All the Code](https://arxiv.org/html/2607.16721)
pruned Qwen3.6-35B-A3B, this model's own base, at 50 percent with no statistically
detectable loss on its primary code benchmark.

## Environment constraints, all verified on this machine

These are not preferences. Each one cost real time to find.

**Serving**

- `enforce_eager=True` is mandatory. CUDA graph capture is numerically broken on
  SM120 for every MoE and attention backend.
- `language_model_only=True` is mandatory. Without it vLLM profiles a
  16,384-token image budget through a vision tower that has zero trained weights,
  and warmup runs for over 16 minutes without finishing.
- `gpu_memory_utilization=0.90`. At 0.95 the engine refuses to start: only
  14.66 of 15.92 GiB is free because the Windows desktop holds about 1.26 GiB.
- Do not use `cpu_offload_gb`. That path dies with an illegal memory access in
  `vllm/model_executor/offloader/uva.py:119`. It is only needed for checkpoints
  too large to fit, which the pruned model is not.

**The model has no vision tower**

The config declares `Qwen3_5MoeForConditionalGeneration`, but the weights are
text-only: 31,333 tensors, every one under `model.language_model.`, zero matching
visual, vision, or patch_embed. The declared tower is randomly initialised on every
load. This caused three separate failures (a warmup hang, a missing-processor load
error, and `Qwen3VLVideoProcessor` failing on absent torchvision), so a release
build should strip the vision config rather than carry it.

**Quantization**

- `NVFP4A16` (weight-only) is **data free**. llm-compressor infers a
  `DataFreePipeline` and the whole job takes 82 seconds.
- `NVFP4` (W4A4) quantizes activations, needs real calibration, and took
  **28.7 minutes** for the same model.
- Both produce the same size: 13.28 vs 13.29 GiB. Activations are never stored, so
  the choice between them is purely speed versus accuracy, with no size cost.
- Pass `processor=tokenizer` to `oneshot`. Otherwise `AutoProcessor` tries to build
  a video processor for the phantom multimodal config and fails.
- Grepping `num_bits: 4` cannot distinguish W4A4 from W4A16. Only
  `input_activations` decides it.

**Pruning**

- reap's save path drops four files the loader needs: `preprocessor_config.json`,
  `video_preprocessor_config.json`, `merges.txt`, `vocab.json`. See
  `scripts/prune/fix_ckpt_files.sh`.
- Router renormalization was silently disabled, because reap gated it on
  `getattr(config, "norm_topk_prob", False)` while Qwen3.5 renormalizes
  unconditionally in the router forward and omits the flag from its config. Fixed
  by asking the adapter instead. Directories named `reap-renorm_true-...` recorded
  the requested value, not the effective one, so the bug was invisible in the logs.
- Changing renormalization invalidates every cached `observations_*.pt`. Re-runs
  need a fresh `--artifacts-dir`, since the aggregate cache is keyed by path and
  will otherwise return a stale hit.

**Tooling traps**

- The lm-eval subcommand is `run`, not `eval`. Worse, `lm_eval eval --help` exits
  **0**, because the top-level parser consumes `--help`, so a naive check passes
  for a subcommand that does not exist.
- Exit codes are unreliable across this stack: lm-eval prints full tracebacks and
  exits 0, and vLLM aborts at teardown with rc=134 after writing valid results.
  Judge every stage by artifacts on disk.
- llm-compressor's REAP does **not** support this architecture. It detects MoE
  layers by duck typing and requires `LinearExperts2D`; Qwen3.5's fused
  `Qwen3_5MoeExperts` fails the check, so all 40 layers are rejected. Pruning uses
  the reap fork; llm-compressor is used only for quantization. See
  `scripts/probes/probe_lc_reap.py`.

## Measured

Cost model on this machine (RTX 5070 Ti, 78 GB usable RAM in WSL):

| operation | cost |
|---|---|
| REAP calibration, 64 samples at 2048 tokens | 57.5 min |
| Prune to 50 percent | ~3 min (2.5 min load, 26 s streaming write) |
| NVFP4A16 quantization | 82 s (data free) |
| NVFP4 W4A4 quantization | 28.7 min (needs calibration) |
| Load 13.28 GiB checkpoint into vLLM | 31 s |

Preliminary and not yet quotable: a first warmup measurement put the weight-only
build at 18.9 tok/s and W4A4 at about 13.3 tok/s, single stream, 512 in and 256 out.
That is the opposite of the expected direction and is being re-measured properly
(see below).

## Open questions and roadmap

**1. Speed is the gating factor.** An agentic coder makes many sequential calls, so
throughput decides whether the model is usable at all. The weight-only build routes
to the `MARLIN` kernel, which dequantizes to bf16 rather than using the FP4 tensor
cores, because `VLLM_CUTLASS` is unavailable to weight-only schemes. W4A4 unlocks
native FP4 but selected `FLASHINFER_CUTLASS`, which prior work on this machine found
to be SM100-oriented. Forcing `VLLM_CUTLASS` is the next experiment.

**2. Accuracy is unmeasured.** Perplexity is not sufficient: the same paper above
reports general perplexity rising 3.5x at the best 50 percent keep point while code
perplexity rises only 1.5x, so perplexity overstates the damage. HumanEval, then
SWE-bench via `mini-swe-agent`, compared against that harness's own leaderboard and
never against the published 69.40, which was measured under a different scaffold.

**3. Rebuild properly.** The current checkpoints were pruned before the
renormalization fix. REAP's ablation implies roughly 0.7 points are recoverable.

**4. Speculative decoding.** MTP is unavailable (`mtp_num_hidden_layers: 0`, despite
several `-MTP-GGUF` repos implying otherwise). N-gram speculation is the realistic
option, reported at about 1.10x on coding workloads.

**5. Healing.** Literature consistently finds one-shot pruning is not the ceiling:
[SlimMoE](https://arxiv.org/html/2506.18349v1) prunes to an intermediate size and
distills to recover, repeating to target. The unpruned model is a natural teacher
since it shares layer, expert, and dimension counts. A 36 GB student with a 69 GB
teacher will not train on this machine, so this is a cloud step or a deliberate skip.

## Honest positioning

The technique is not novel and should not be claimed as such. Verified against the
Hugging Face Hub on 2026-08-17:

- REAP combined with NVFP4 on `qwen3_5_moe` already exists
  (`rene98c/Qwen3.5-397B-A17B-REAP-28-NVFP4`, March 2026, 23.1K downloads).
- REAP on this specific model exists as GGUF
  (`gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF`).

What is unclaimed is a **vLLM-servable KAT-Coder that is genuinely usable in 16 GB**.
The bar to beat is Devstral-2 22B at 52.3 percent agentic in a comparable footprint.

A separate measurement from this work, on the stability of REAP's expert ranking
under different calibration draws, is tracked independently.

## Layout

```
scripts/prune/     REAP pruning, calibration stability, the renormalization fix
scripts/quantize/  NVFP4A16 and NVFP4 W4A4 builds via llm-compressor
scripts/eval/      paired evaluation, McNemar, resolution diagnostics
scripts/bench/     A/B latency via vllm bench, serving smoke tests
scripts/probes/    cheap precondition checks that run before expensive jobs
tasks/             lm-eval task definitions
```

## Measurement conventions

Numbers here follow a few rules, learned the hard way:

- Report median and range over at least five separate process invocations. Variance
  lives between invocations, not inside them.
- Interleave A/B runs rather than blocking them, so thermal drift and run order
  cannot masquerade as the effect being measured.
- Discard a warmup run. A cold compile cache measures the compiler.
- Report the resolution diagnostic alongside any null result. A non-significant
  difference from an underpowered test is not evidence of no difference.
- Use the standard tool. `vllm bench` rather than a hand-rolled script, after one
  such script reported 6.5 tok/s for a workload the official tool measured at 111.3.
