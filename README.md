# kat-coder-16gb

Making `Kwaipilot/KAT-Coder-V2.5-Dev` (69.40 SWE-bench Verified) run as a usable
local agentic coding model inside 16 GB of consumer VRAM, on an RTX 5070 Ti (SM120).

Pipeline: REAP expert pruning at 50 percent, then NVFP4 quantization, served by vLLM.

## Results

| metric | value | conditions |
|---|---|---|
| **Size** | **12.45 GiB** | REAP 50% + NVFP4A16, vision-stripped |
| **Speed** | **149.5 tok/s** median, n=5 | 512 in / 256 out, batch 1, CUDA graphs (PIECEWISE) |
| **HumanEval+** | **89.0%** [83.3, 92.9] | greedy, instruct framing, 164 problems |
| **MBPP+** | **90.5%** [87.1, 93.0] | greedy, instruct framing, 378 problems |
| **SWE-bench Verified** | **40.0%** (20/50 resolved) | mini-swe-agent bash-only, 32K context ceiling; 27/50 empty patch (18 from hitting the ceiling) |
| **Load time** | 28.9 s | CUDA graphs enabled, no CPU offload |

Release candidate: `kat-50pct-nvfp4a16-renorm-stripped`. Renorm-corrected,
vision-free, loads on SM120 with no CPU offload. NVFP4 compute is numerically
correct on this architecture. No pad collapse, no NaN.

SWE-bench Verified is below the competitive bar (Devstral Small 2512, 56.4%
under the same scaffold) — see Honest positioning. Over half the empty-patch
failures trace to the 32K context ceiling rather than the model failing the
task; context-window work is in progress on `feat/optimize-vllm-and-agent-config`.

## Status

| stage | status |
|---|---|
| REAP 50% prune (Qwen3.5 MoE support added to reap fork) | done |
| Router renormalization fix | done, committed for upstream |
| NVFP4A16 quantization (data-free, 82 s) | done |
| Vision tower stripped | done (no trained weights existed) |
| Speed benchmarked (149.5 tok/s, n=5) | done |
| CUDA graphs (7.4x over eager) | done |
| Agentic serving config (prefix caching, 45x) | done |
| HumanEval+ / MBPP+ accuracy | done (89.0% / 90.5%) |
| SWE-bench Verified via mini-swe-agent | done — 20/50 = 40.0%, 18 CWE (32K ceiling); 60% experiment on `feature/60pct-prune` |
| Release checkpoint on Hugging Face | not yet published |

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

Working agentic config:

```
--max-model-len 32768 --max-num-seqs 2 --gpu-memory-utilization 0.92
--kv-cache-dtype fp8 --enable-prefix-caching --max-num-batched-tokens 4096
--reasoning-parser qwen3 --language-model-only
--compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}'
```

The KV budget fluctuates 0.49-1.41 GiB with the Windows desktop's VRAM.
Never set `max_model_len` near a measured ceiling; 32768 survives the worst case.
See `scripts/swebench/README.md` for the full measured table.

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
`model.language_model.`. The declared tower is randomly initialised on every load.
The release candidate strips the vision config.

**Quantization**

- `NVFP4A16` (weight-only) is **data free** and takes 82 seconds.
- `NVFP4` (W4A4) needs real calibration and takes 28.7 minutes.
- Both produce the same size. Weight-only wins on speed (1.32x) and accuracy safety.
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
scripts/quantize/    NVFP4A16 and NVFP4 W4A4 builds via llm-compressor
scripts/eval/        HumanEval+/MBPP+, paired evaluation, McNemar
scripts/bench/       A/B latency via vllm bench, serving smoke tests
scripts/swebench/    SWE-bench Verified via mini-swe-agent (agentic evaluation)
scripts/probes/      cheap precondition checks that run before expensive jobs
tasks/               lm-eval task definitions
docs/                environment setup guide
```

## Measurement conventions

- Report median and range over at least five separate process invocations.
- Interleave A/B runs rather than blocking them.
- Discard a warmup run.
- Report the resolution diagnostic alongside any null result.
- Use the standard tool (`vllm bench`, `lm-eval-harness`, `mini-swe-agent`).

## License

Apache 2.0, matching `reap` and `llm-compressor` so the router renormalization
fix can be offered upstream without a licence mismatch.
