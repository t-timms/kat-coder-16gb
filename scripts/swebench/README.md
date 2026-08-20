# SWE-bench Verified evaluation

Running `KAT-Coder-V2.5-Dev` (pruned + quantized) on SWE-bench Verified via
`mini-swe-agent`, the SWE-bench authors' own harness that powers the standardised
bash-only leaderboard.

## Prerequisites

- Docker Engine in WSL (`wsl -u root` then install, no sudo needed from Windows)
- `~/swebench-env` with mini-swe-agent 2.4.6 and swebench 5.0.1
- `~/vllm-env` with vLLM 0.20.2 source build (SM120 support)
- `~/models/kat-50pct-nvfp4a16-renorm-stripped` (12.45 GiB release candidate)

## Architecture

```
run_pilot_all.sh N [output_dir]
  |
  +-- starts vLLM with agentic config (PIECEWISE graphs, prefix caching)
  +-- runs preflight_litellm.py (validates through litellm, not curl)
  +-- runs mini-swe-agent rollout (N instances, 2 workers)
  +-- captures prefix cache hit counters
  +-- reports predictions
  +-- tears down vLLM on exit (trap)
```

## Critical flags

The serve script includes flags that are NOT optional:

- `--enable-auto-tool-choice --tool-call-parser qwen3_xml`: mini-swe-agent's
  litellm model class sends `tool_choice: auto` with `tools=[BASH_TOOL]`
  unconditionally. Without these flags, every LM call returns 400.
- `--enable-prefix-caching --max-num-batched-tokens 4096`: prefix caching is 45x
  on replayed history, but the engine asserts if `max_num_batched_tokens` (default
  2048) is below the Mamba block_size (2096).
- `--reasoning-parser qwen3`: routes `<think>` traces to `reasoning_content`,
  keeping `content` clean for action parsing and out of replayed history.

## Usage

```bash
# Run 5 instances (pilot)
bash scripts/swebench/run_pilot_all.sh 5

# Run 50 instances (full evaluation)
bash scripts/swebench/run_pilot_all.sh 50

# Grade the results with the official harness
bash scripts/swebench/grade_pilot.sh
```

`max_num_seqs` is 8 (tested 2026-08-19, 1.86x concurrency at 32K context,
throughput only — does not affect the score). To try the unvalidated
context-budget config instead of the default, aimed at the 18
ContextWindowExceeded failures below:

```bash
KAT_CONFIG=kat_overrides_context_managed.yaml bash scripts/swebench/run_pilot_all.sh 50
```

`$CFGDIR` (`~/kat_swebench`) must have this file alongside `kat_overrides.yaml`
before running — copy it over from `scripts/swebench/` if it isn't there yet.

## Results

**50-instance Verified run** (50% REAP + NVFP4A16 model, pre-optimization
agent config, 2026-08-19):

| metric | value |
|---|---|
| resolved | **20/50 = 40.0%** |
| completed (valid patch produced) | 22 |
| resolved of completed | 20/22 = 90.9% |
| ContextWindowExceeded | 18 (32K ceiling) |
| LimitsExceeded | 9 |
| garbage/invalid patch | 1 (of 23 generated) |

Result file: `results/hosted_vllm__kat-16gb.kat-coder-16gb-50.json`. The run
must be disclosed as 32K-step-limited (see Context constraint below). A 60%
sparsity experiment runs on `feature/60pct-prune` with the identical agent
config; results will be recorded here when graded.

## Context constraint

The 32K context window is the safe ceiling. The KV budget fluctuates 0.49-1.41 GiB
with the Windows desktop's VRAM, so `max_model_len` near a measured ceiling will
fail intermittently. Devstral Small averages 86.9 LM calls per instance; many
instances will hit the context limit. The run must be disclosed as step-limited.

The step limit in `kat_overrides.yaml` is 40. The pilot measured ~590 tokens/step
average, so 40 steps x 590 = ~23,600 tokens of history, which fits in 32K with
room for generation. But the context grows across steps, and later steps may
trigger ContextWindowExceeded.

## Grading

Uses the official `swebench.harness.run_evaluation` with `SWE-bench/SWE-bench_Verified`
(the new org; `princeton-nlp/` is the old schema and does not work with swebench 5.0.1).

Docker images are cached locally. First instance of a new repo pulls ~3.5-4.2 GiB;
subsequent instances of the same repo reuse layers (226-1922 MiB). SWE-bench Verified
is 12 distinct repos (231 of 500 are django), so ~50 instances is ~65 GB of images.

## Preflight

`preflight_litellm.py` validates the serving config through litellm with the same
BASH_TOOL definition mini-swe-agent uses. This catches the auto-tool-choice error
that a curl-based preflight missed. The script aborts before a single Docker
container starts if anything is wrong.

## Files

| file | purpose |
|---|---|
| `run_pilot_all.sh` | serve + rollout + teardown in one process |
| `grade_pilot.sh` | official SWE-bench grading harness |
| `serve_kat.sh` | standalone server (for interactive use) |
| `preflight_litellm.py` | validates litellm tool calling path |
| `analyze_pilot.py` | analyzes rollout trajectories |
| `kat_overrides.yaml` | model config layered on stock swebench.yaml (default) |
| `kat_overrides_context_managed.yaml` | opt-in, unvalidated: shorter step limit/max_tokens/observation truncation, targets the CWE failures |
| `registry.json` | litellm model registry for local vLLM |
