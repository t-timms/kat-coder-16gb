# Roadmap

Status snapshot as of 2026-08-21. Not a changelog (see `CHANGELOG.md` for what
shipped) — this is where the project is headed and why, kept current rather
than historical.

## Done

- REAP 50% expert-prune + NVFP4A16 checkpoint published:
  [`Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16),
  12.45 GiB, fits a single 16 GB SM120 card. Accuracy reproduced on the shipped
  weights (HumanEval+, MBPP+ both inside published confidence intervals).
- W4A4 measured against the shipped A16 checkpoint (accuracy cost small,
  -0.6pp HumanEval+ / -0.3pp MBPP+, one problem in each, per `CHANGELOG.md`;
  not shipped — see Next Major below).
- SWE-bench Verified pilot baseline: **20/50 (40.0%)** at 32,768-token context,
  18 of 30 failures were `ContextWindowExceeded`, not capability failures (20 of
  22 patch-producing instances resolved).
- Context-window fix measured and 1-instance-validated: `MAXLEN=49152
  MAXSEQS=2` gives 98,304 total KV tokens — both `--workers 2` rollout workers
  hold a full 49,152-token context simultaneously, zero preemption. Config
  lives in `scripts/swebench/kat_overrides_sota.yaml`.
- 2026-08-21 SOTA audit (`docs/optimization_research_2026-08-21.md`): vLLM
  0.27.1, the `lna-lab` community SM120 patches, and NVFP4 KV cache all
  investigated and rejected for the current checkpoint, with evidence. No
  changes needed to the serving config as a result — it was already correct.

## Next (this session's actual deliverable — not yet run)

**Launch the 50-instance SWE-bench Verified pilot** at the validated
`MAXLEN=49152 MAXSEQS=2` config, to see whether raising context resolves the
18 `ContextWindowExceeded` failures and moves the score past 40.0%.

```bash
cd ~/kat-coder-16gb/scripts/swebench
MAXLEN=49152 MAXSEQS=2 KAT_CONFIG=kat_overrides_sota.yaml \
  bash run_pilot_all.sh 50 ~/swebench_sota
```

Grade afterward with `bash grade_pilot.sh` against `~/swebench_sota`. Runs
several hours unattended (2 Docker-backed rollout workers) — launch when there
is a multi-hour block free, close Chrome first (last discretionary VRAM
holder), and confirm `nvidia-smi` shows a clean GPU beforehand.

**This step is explicitly gated on being told to launch it** — do not start
the 50-instance run automatically; wait to be asked.

## Next major project: W4A4 re-quantization

The real remaining lever, not a same-session change. From
`docs/optimization_research_2026-08-21.md` §2: our current checkpoint is
NVFP4A16 (weight-only), and real FP4 tensor-core kernels require FP4×FP4
(weight+activation) inputs — there is no hardware path for a weight-only
scheme to reach them, on any GPU, so MoE experts always fall back to Marlin
regardless of vLLM version or device patches. Already measured on this exact
model (2026-08-20 entry): W4A4 reaches native kernels, ~+31% throughput,
-0.6pp HumanEval+ / -0.3pp MBPP+ accuracy cost (one problem in each) — small
relative to the ~38pp INT4-era collapse the older literature assumed (NVFP4's
per-16-block scaling holds up much better).

Scope for that project:

1. Re-quantize the pruned checkpoint with `scheme: NVFP4` (W4A4) instead of
   `NVFP4A16`, using `scripts/quantize/quantize_kat_w4a4.py` (already exists,
   unused for the shipped release).
2. Re-run the accuracy suite (HumanEval+, MBPP+) on the W4A4 checkpoint to
   confirm the measured delta holds at this specific prune ratio and seed.
3. Revisit the `lna-lab/blackwell-geforce-nvfp4-gemm` patch set
   (`~/blackwell-patches`, cloned 2026-08-21) — irrelevant to A16 (see above)
   but potentially relevant to W4A4, since a W4A4 checkpoint will actually
   reach the CUTLASS/FlashInfer MoE dispatch these patches touch. Re-check
   whether the device-family gates they patch are still needed once a real
   W4A4 `QuantKey` is in play, or whether upstream has closed those gaps too.
4. Account for the 833s FlashInfer JIT first-load cost measured 2026-08-20 —
   decide whether a warm-container / pre-JIT strategy makes this acceptable
   for a serving deployment, since that was the original blocker for shipping
   W4A4, not the accuracy cost.
5. Decide publish strategy: replace the A16 release, or publish alongside it
   as a speed-optimized variant with its own accuracy disclosure.

The `~/vllm-env-027` / `~/vllm-src-027` build (vLLM 0.27.1, SM120, torch
2.13.0+cu130) is left on disk as a starting point if this project wants to
revisit the two SM120-specific decode-throughput fixes noted in
`docs/optimization_research_2026-08-21.md` §1 — those didn't help the A16
checkpoint but weren't tested against W4A4's different kernel path.

## Longer-horizon / not scheduled

- GGUF quant for reach (llama.cpp/Ollama/LM Studio compatible) — current
  model has 74 HF downloads (verified 2026-08-21); NVFP4A16 needs vLLM +
  Blackwell specifically, which caps the addressable audience. Verified
  2026-08-21 (prior "205 experts, REAP" note in this doc was wrong and has
  been removed): `gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF` (45.3K downloads) is
  **not** REAP-pruned — it's the full, unpruned `Kwaipilot/KAT-Coder-V2.5-Dev`
  quantized to GGUF, with an MTP head grafted from Qwen3.6-35B-A3B (their
  README independently confirms our own finding: KAT ships
  `mtp_num_hidden_layers: 0`, no native draft head). Not a fair architecture
  comparison to our 50%-pruned checkpoint; if we ship a GGUF it would be the
  first REAP-pruned one for this base model.
- Base-model swap to Ornith-1.5-35B-A3B — tracked separately in the
  `sota-ornith-build` branch of the private 60%-experiment repo, not this one.
