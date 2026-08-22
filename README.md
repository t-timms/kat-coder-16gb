# kat-coder-nvfp4

Making `Kwaipilot/KAT-Coder-V2.5-Dev` (69.40 SWE-bench Verified) run as a usable
local agentic coding model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120).

Pipeline: REAP expert pruning at 50 percent, then NVFP4 quantization, served by vLLM.

Two checkpoints are published: [`KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16)
(weight-only, this release's default — see Results below) and
[`KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4)
(weights + activations, native FP4 kernels — see [W4A4](#w4a4-an-alternative-quantization-strategy) below).

## Results

| metric | value | conditions |
|---|---|---|
| **Size** | **12.45 GiB** | REAP 50% + NVFP4A16, vision-stripped; 3 shards + index |
| **Speed** | **149.5 tok/s** median, n=5 | 512 in / 256 out, batch 1, CUDA graphs (PIECEWISE) |
| **HumanEval+** | **89.0%** [83.3, 92.9] | greedy, instruct framing, 164 problems; reproduced at 90.9% on the released checkpoint |
| **MBPP+** | **90.5%** [87.1, 93.0] | greedy, instruct framing, 378 problems; reproduced at 89.9% on the released checkpoint |
| **SWE-bench Verified** | **52.0%** (26/50 resolved) | mini-swe-agent bash-only, 49K context ceiling; 17/50 empty patch, all from hitting the ceiling |
| **Load time** | 28.9 s | CUDA graphs enabled, no CPU offload |

Released checkpoint: [`Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16).
Renorm-corrected, vision-free, loads on SM120 (compute capability 12.0) with no
CPU offload. NVFP4A16 is weight-only, so no FP4 arithmetic is required or used:
vLLM serves it through the Marlin NVFP4 kernel, which decodes the 4-bit weights
and computes in bf16. Numerically clean — no pad collapse, no NaN.

SWE-bench Verified is still below the competitive bar (Devstral Small 2512,
56.4% under the same scaffold — see Honest positioning) but closed most of
the gap: raising both the context ceiling (32K→49K tokens, `max_model_len
49152`, `max_num_seqs 2`) and the agent step limit (40→65, paired changes in
`kat_overrides_sota.yaml`) moved the score from 40.0% to 52.0%. The
context-window failure rate barely moved (34% vs. the prior 36%) — the real
lever was the step limit: the old config's 9 `LimitsExceeded` failures
(instances that ran out of agent turns before finishing, a *different*
failure mode than hitting the context ceiling) are gone entirely at the
raised limit, so 10 more instances (22→32) reached a real completion
attempt instead of running out of steps first. Those newly-reachable
instances — the ones that previously failed purely from the step ceiling —
resolve at a lower rate than instances that were already completing
(81.25% resolved-of-completed at 49K vs. 90.9% at 32K, consistent with them
being the harder, longer problems), but enough resolved anyway that the net
count still rose (20→26). This is now the default agentic config — see
Status below.

## Status

| stage | status |
|---|---|
| REAP 50% prune (Qwen3.5 MoE support added to reap fork) | done |
| Router renormalization fix | done, committed for upstream |
| NVFP4A16 quantization (data-free, 82 s) | done |
| Vision tower removed — declaration and weights | done |
| Speed benchmarked (149.5 tok/s, n=5) | done |
| CUDA graphs (7.4x over eager) | done |
| Agentic serving config (prefix caching, 45x) | done |
| HumanEval+ / MBPP+ accuracy | done (89.0% / 90.5%) |
| SWE-bench Verified via mini-swe-agent | done — 26/50 = 52.0% at 49K context (`kat_overrides_sota.yaml`, now the default); 40.0% at the original 32K ceiling, kept below as the prior baseline. The 60% sparsity comparison runs in [a separate repository](https://github.com/t-timms/kat-coder-16gb-60pct-experiment) |
| Rollout throughput (`max_num_seqs` 2→8) | done, tested — 1.86x concurrency, no score impact |
| Context ceiling raised 32K→49K (`max_model_len`, `max_num_seqs` 8→2) | done, validated on the full 50-instance pilot — see Results above |
| Context-*budget* experiment (opt-in, reduces `max_tokens` instead of raising the ceiling) | scaffolded, still unvalidated, a different lever than the one above — see `kat_overrides_context_managed.yaml` |
| `presence_penalty`/`top_k` added, completing the model's documented sampling recommendation | done, config only — tested single-instance with repeated draws (suppressed a real repetition-loop failure), **full-pilot re-validation not yet run** — see `kat_overrides_sota.yaml`'s header comment |
| Release checkpoint on Hugging Face | published — [`Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16) |
| W4A4 (native FP4 kernels) alternative build | published, see below |
| GPTQ-based NVFP4A16 requantization (same size, different rounding algorithm) | scoped and scripted (`scripts/quantize/quantize_kat_gptq.py`), **not yet run** — see `ROADMAP.md`'s dated entry for the verification behind this and what would make it worth shipping |

## W4A4: an alternative quantization strategy

Same pruned base, quantized differently: activations to 4 bits as well as
weights (`NVFP4`, i.e. W4A4), instead of this release's weight-only
NVFP4A16. That reaches native FP4×FP4 tensor-core kernels instead of
Marlin's dequant-to-bf16 fallback. Measured both in isolated eager mode and
under the same PIECEWISE CUDA-graph configuration this repo actually serves
with, 5 interleaved runs per arm, median + range:

| | HumanEval | HumanEval+ | MBPP+ | decode (PIECEWISE, production-representative) |
|---|---:|---:|---:|---:|
| A16 (published, above) | 95.7% | 90.9% | 89.9% | 142.5 tok/s |
| W4A4 ([published separately](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4-W4A4)) | 92.07% | 89.02% | 91.01% | 119.2 tok/s |

On this hardware, A16 is faster (W4A4 at 0.84x) with a mixed accuracy
picture (lower on HumanEval/HumanEval+, slightly higher on MBPP+) — both
throughput numbers are well into comfortably-interactive territory, this
just documents which build we default to and why. Full writeup, including
the memory-margin and kernel-selection findings along the way, in
`ROADMAP.md`'s RESULT section and `docs/optimization_research_2026-08-21.md`.
Only single-stream (batch=1) decode was measured; the tradeoff at higher
concurrency is untested.

## Quickstart

Environment setup, including why three separate Python environments are required,
is in [docs/environment.md](docs/environment.md). Expect to need an RTX 5070 Ti or
another SM120 card, about 250 GB of disk, and 80 GB allocated to WSL2.

**1. Check preconditions before spending hours on a run**

```bash
bash scripts/probes/check_quant_preconditions.sh
```

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

# build the shippable checkpoint (removes the untrained vision tower)
~/quant-env/bin/python scripts/release/build_release_candidate.py
```

**4. Confirm it serves and produces real code**

```bash
~/vllm-env/bin/python scripts/bench/smoke_pruned_nvfp4.py
```

**5. Benchmark**

```bash
bash scripts/bench/bench_ab.sh 5
~/vllm-env/bin/python scripts/bench/bench_ab_analyze.py
```

**6. Run SWE-bench (agentic evaluation)**

```bash
# Prerequisite: Docker Engine in WSL, mini-swe-agent + swebench installed
bash scripts/swebench/run_pilot_all.sh 50    # ~2-3 hours for 50 instances
bash scripts/swebench/grade_pilot.sh         # official SWE-bench harness
```

The serve + rollout + teardown are combined in one script because starting the
server from a separate invocation reports READY and then dies when that invocation
exits. See `scripts/swebench/README.md` for the full agentic pipeline docs.

Defaults are `MAXLEN=49152 MAXSEQS=2 KAT_CONFIG=kat_overrides_sota.yaml`. The
52.0% figure above was measured on this config as it stood on 2026-08-22;
`kat_overrides_sota.yaml` has since had two missing sampling parameters added
(`presence_penalty`, `top_k`, completing the base model's own documented
recommendation — see the file's own header comment for what this does and
doesn't fix), tested on one instance with repeated draws but **not yet
re-validated on a full 50-instance pilot**. Treat 52.0% as the last
full-pilot-validated number, not a live reflection of the current default
config, until that re-validation runs. To reproduce the original 32K/40.0%
baseline instead, run `MAXLEN=32768 MAXSEQS=8 KAT_CONFIG=kat_overrides.yaml bash
scripts/swebench/run_pilot_all.sh 50`. A third, still-unvalidated config
(reduces `max_tokens` instead of raising the context ceiling) is available via
`KAT_CONFIG=kat_overrides_context_managed.yaml`.

**7. Run HumanEval / MBPP+ (non-agentic accuracy)**

```bash
bash scripts/eval/eval_suite.sh              # ~8 min for 706 problems
~/swebench-env/bin/python scripts/eval/read_scores.py  # Wilson CIs
```

## Why 50 percent

Forced by arithmetic, not chosen:

| variant | size | fits 16 GB |
|---|---|---|
| bf16 base | 69.3 GB | no |
| NVFP4, unpruned | 21.9 GB | no |
| REAP 25% + NVFP4 | ~16-17 GB | no, not once KV cache is counted |
| **REAP 50% + NVFP4** | **12.45 GiB** | **yes** |

Supporting evidence: [Half the Experts, All the Code](https://arxiv.org/html/2607.16721)
pruned Qwen3.6-35B-A3B, this model's own base, at 50 percent with no statistically
detectable loss on its primary code benchmark.

## Agentic serving (not the same as benchmark serving)

The 149.5 tok/s benchmark config serves only **14,672 tokens of context** and
cannot run an agent. The agentic config trades 7% speed for 4.4x more context:

| cudagraph_mode | tok/s | max context |
|---|---:|---:|
| FULL_AND_PIECEWISE | 149.5 | 14,672 |
| **PIECEWISE** | **139.4** | **64,976** |
| eager | 19.9 | 148,816 |

**Prefix caching is the single biggest agentic lever: 45x.** An agent replays its
whole history every step. Measured on a 13,130-token history:

| | cold | warm (+1 step) |
|---|---:|---:|
| caching OFF | 31.25 s | 30.74 s |
| **caching ON** | 9.39 s | **0.21 s** |

Working agentic config, validated on a full 50-instance run (0 infrastructure
failures, 0 crashes — see Results above):

```
--max-model-len 49152 --max-num-seqs 2 --gpu-memory-utilization 0.92
--kv-cache-dtype fp8 --enable-prefix-caching --max-num-batched-tokens 4096
--reasoning-parser qwen3 --language-model-only
--compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}'
```

The KV budget fluctuates with the Windows desktop's VRAM. `max_model_len 32768`
is the more conservative fallback if this ceiling ever proves too tight on a
different machine — it was the original validated config (40.0%) before the
49K ceiling replaced it as the default. See `scripts/swebench/README.md` for
the full measured table.

## Environment constraints, all verified on this machine

**Serving**

- **CUDA graphs work.** They were long believed numerically broken on SM120, and
  that belief is wrong for this model on vLLM 0.20.2. Three settings are required:
  - `max_num_seqs=2` (for agentic) or `4` (for benchmark). The default 256
    exceeds available Mamba cache blocks on this hybrid architecture.
  - `cudagraph_capture_sizes=[1,2]` (agentic) or `[1,2,4,8]` (benchmark).
  - Do **not** set `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` with graphs on.
- `language_model_only=True` is mandatory. Without it vLLM profiles a 16K-token
  image budget through a vision tower with zero trained weights.
- `gpu_memory_utilization=0.92`. Higher values fail because the Windows desktop
  holds 0.5-1.0 GiB of VRAM that fluctuates.
- Do not use `cpu_offload_gb`. That path dies with an illegal memory access.

**The model has no vision tower.** The config declares
`Qwen3_5MoeForConditionalGeneration`, but all 31,333 weight tensors are under
`model.language_model.`. Transformers materialises the declared tower as random
values on every load, and anything that saves the model afterwards writes those
values out — 333 untrained BF16 tensors, 0.83 GiB. The quantizer's `re:visual.*`
rule keeps them at full precision, and removing `vision_config` strips the
declaration without touching them.
`scripts/release/build_release_candidate.py` removes the declaration and the tensors
together, which is the difference between a 13.28 GiB build artifact and the
12.45 GiB released checkpoint. See [docs/checkpoint_composition.md](docs/checkpoint_composition.md).

**Quantization**

- `NVFP4A16` (weight-only) is **data free** and takes 82 seconds.
- `NVFP4` (W4A4) needs real calibration and takes 28.7 minutes.
- Both produce the same size: 12.4532 GiB (W4A4) against 12.4512 GiB (A16), after
  the vision tower is removed from each.
- **The scheme decides which kernel you get.** vLLM routes NVFP4A16 to
  `MarlinNvFp4LinearKernel`, logging that the GPU has no native FP4 support, and
  routes W4A4 to `FlashInferCutlassNvFp4LinearKernel` with no such warning. The
  native FP4 GEMM on SM120 is reachable only by a scheme that also quantizes
  activations; weight-only cannot reach it at any setting.
- **W4A4 costs almost nothing in accuracy here**, which is not what the prior
  literature predicts. Same pruned source, same ignore list, same greedy decoding:

  | benchmark | NVFP4A16 | NVFP4 W4A4 | delta | n |
  |---|---:|---:|---:|---:|
  | HumanEval | 95.7% | 93.3% | -2.4 pp (157 -> 153) | 164 |
  | HumanEval+ | 90.9% | 90.2% | -0.6 pp (149 -> 148) | 164 |
  | MBPP+ | 89.9% | 89.7% | -0.3 pp (340 -> 339) | 378 |

  QSpec measured W4A4 losing 38.73% on HumanEval where W4A16 barely moved. That
  did not reproduce: the gap here is one problem on both EvalPlus benchmarks. The
  QSpec result is INT4-era, and NVFP4's per-16 block scaling with FP8 scales is a
  far better conditioned format. All three deltas are negative, so a small real
  cost is likely rather than pure noise, but every interval overlaps heavily.
  Raw results in `results/eval-w4a4/`.
- W4A4's first load cost **833 s** against roughly 40 s for A16, because FlashInfer
  JIT-compiles its kernels; the cache persists, so later loads are normal. Relative
  serving *throughput* is still unmeasured - that needs `scripts/bench/bench_ab.sh`,
  and the smoke test is not a benchmark.
- Pass `processor=tokenizer` to `oneshot` to avoid the phantom video processor.

**Pruning**

- reap's save path drops four files the loader needs. See
  `scripts/prune/fix_ckpt_files.sh`.
- Router renormalization was silently disabled. Fixed by asking the adapter.
  Changing renormalization invalidates cached `observations_*.pt`.
- llm-compressor's REAP does **not** support this architecture (requires
  `LinearExperts2D`; Qwen3.5's fused experts are not). Pruning uses the reap
  fork; llm-compressor is used only for quantization.

**Tooling traps**

- The lm-eval subcommand is `run`, not `eval`. `lm_eval eval --help` exits 0
  without exercising the subcommand.
- Exit codes are unreliable: lm-eval prints tracebacks and exits 0; vLLM
  aborts at teardown with rc=134 after writing valid results.
- Judge every stage by artifacts on disk, not exit codes.

## Measured costs (RTX 5070 Ti, 78 GB usable RAM in WSL)

| operation | cost |
|---|---|
| REAP calibration, 64 samples at 2048 tokens | 57.5 min |
| Prune to 50 percent | ~3 min |
| NVFP4A16 quantization | 82 s (data free) |
| NVFP4 W4A4 quantization | 28.7 min (needs calibration) |
| Load 12.45 GiB checkpoint into vLLM | 28.9 s |
| HumanEval+ + MBPP+ (706 problems) | ~8 min |
| SWE-bench 50 instances (rollout + grade) | ~2-3 hours |

## Honest positioning

The technique is not novel and should not be claimed as such. Verified against the
Hugging Face Hub on 2026-08-17:

- REAP combined with NVFP4 on `qwen3_5_moe` already exists
  (`rene98c/Qwen3.5-397B-A17B-REAP-28-NVFP4`, March 2026, 23.1K downloads).
- REAP on this specific model exists as GGUF
  (`gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF`).

What is unclaimed is a **vLLM-servable KAT-Coder that is genuinely usable in 16 GB**.
The bar to beat is Devstral Small (2512) at 56.4% under the same mini-swe-agent
bash-only scaffold (SWE-bench/experiments, v1.17.2, 86.9 LM calls/instance).

## Layout

```
scripts/prune/       REAP pruning, calibration stability, the renormalization fix
scripts/quantize/    NVFP4A16 (RTN and GPTQ variants) and NVFP4 W4A4 builds via llm-compressor
scripts/release/     assemble and verify the shippable checkpoint
scripts/eval/        HumanEval+/MBPP+, paired evaluation, McNemar
scripts/bench/       A/B latency via vllm bench, serving smoke tests
scripts/swebench/    SWE-bench Verified via mini-swe-agent (agentic evaluation)
scripts/probes/      cheap precondition checks that run before expensive jobs
tasks/               lm-eval task definitions
docs/                environment setup guide, SOTA/optimization research notes
```

## Measurement conventions

- Report median and range over at least five separate process invocations.
- Interleave A/B runs rather than blocking them.
- Discard a warmup run.
- Report the resolution diagnostic alongside any null result.
- Use the standard tool (`vllm bench`, `lm-eval-harness`, `mini-swe-agent`).

## License

Apache 2.0, inherited from the base model `Kwaipilot/KAT-Coder-V2.5-Dev` and
matching `reap` and `llm-compressor`, so the router renormalization fix can be
offered upstream without a licence mismatch.
