# Roadmap

Status snapshot as of 2026-08-21. Not a changelog (see `CHANGELOG.md` for what
shipped) — this is where the project is headed and why, kept current rather
than historical.

## Done

- REAP 50% expert-prune + NVFP4A16 checkpoint published:
  [`Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16`](https://huggingface.co/Ttimms/KAT-Coder-V2.5-Dev-REAP-50-NVFP4A16),
  12.45 GiB, fits a single 16 GB SM120 card. Accuracy reproduced on the shipped
  weights (HumanEval+, MBPP+ both inside published confidence intervals).
- W4A4 measured against the shipped A16 checkpoint (accuracy cost small: per
  `README.md`'s full table, -2.4pp HumanEval, -0.6pp HumanEval+, -0.3pp MBPP+
  — one problem apiece on the two EvalPlus sets; not shipped — see Next Major
  below).
- SWE-bench Verified pilot baseline: **20/50 (40.0%)** at 32,768-token context.
  The 20/50 total and the 22-completed count are independently re-verified
  (2026-08-21) against the raw committed grading artifact,
  `results/hosted_vllm__kat-16gb.kat-coder-16gb-50.json`. The specific
  failure-reason breakdown — 18 of the 30 non-resolved instances attributed to
  `ContextWindowExceeded` — is stated consistently across five pre-existing
  repo files (README, CHANGELOG, HF_MODEL_CARD, both swebench config yamls)
  but its raw per-instance evidence (an `exit_statuses_*.yaml`, the format
  this harness uses to record failure reasons — confirmed present for other,
  smaller runs, e.g. `~/swebench_validate1/exit_statuses_*.yaml`) no longer
  exists on disk for this specific 50-instance run. Carried forward as
  previously reported, not independently re-verified tonight.
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
reported `ContextWindowExceeded` failures (see sourcing caveat under Done
above) and moves the score past 40.0%.

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
model (2026-08-20 entry): W4A4 reaches native kernels (`FlashInferCutlassNvFp4LinearKernel`
vs A16's `MarlinNvFp4LinearKernel` dequant fallback), same checkpoint size
(12.4532 vs 12.4512 GiB), -2.4pp HumanEval / -0.6pp HumanEval+ / -0.3pp MBPP+
accuracy cost (README's full table) — small relative to the ~38pp INT4-era
collapse QSpec reported (NVFP4's per-16-block scaling holds up much better).

**Correction 2026-08-21 (self-audit while re-reading this section against
README.md):** this section previously said "~+31% throughput" was "already
measured." It was not, and README.md's own "Measured costs" section says so
explicitly: *"Relative serving throughput is still unmeasured - that needs
`scripts/bench/bench_ab.sh`, and the smoke test is not a benchmark."* Traced
the "~31%" figure to its actual origin: a rough, uncited expectation written
in `scripts/quantize/quantize_kat.py`'s docstring (the *A16* script) as
motivation for building the weight-only checkpoint first, dated before any
W4A4 checkpoint existed to measure. It is not a citation to a specific paper
and not a local measurement — carrying it forward as "measured" was an
error. What IS measured and real: the kernel-path difference itself (Marlin
dequant-fallback vs native FlashInfer CUTLASS FP4 tensor-core GEMM) and the
accuracy deltas above. The actual throughput number is still open and is
exactly what `scripts/bench/bench_ab.sh` is for — run it on the new build
before quoting any percentage to anyone.

**Launch-prep audit done 2026-08-21 (no GPU/CPU compute spent — see
`docs/optimization_research_2026-08-21.md` §4 for full detail):** the recipe,
the environment, and the two open risk items below were checked against
upstream sources and this box's actually-installed packages/source before
committing CPU-hours to a calibration run. Summary: recipe already matches
upstream's own Qwen3.5-MoE NVFP4 example almost exactly (one deliberate
addition below); the SM12x NaN-risk bug (vLLM #35947/#37725) is confirmed
already fixed in this box's `~/vllm-src` (tag v0.26.0, built four months
after the fix merged); all four "Critical priority" lna-lab patches relevant
to MoE FP4 on SM120 are confirmed already present in the installed
`flashinfer==0.6.14`/`vllm-src` — verified by grep, not inferred (step 3
below, resolved without applying anything). One recipe change made:
`MAX_SEQ` raised 2048 → 4096 to match upstream's example and this project's
own 49,152-token serving window (roughly doubles CPU calibration wall-clock;
78 GiB free RAM confirmed sufficient). What's still genuinely unverified:
an end-to-end W4A4 MoE forward pass has never run on this card — the
individual kernel paths check out on paper but the specific combination
(runtime activation quant + FlashInfer CUTLASS MoE + piecewise CUDA graphs)
is untested here.

## RESULT (2026-08-21, GPU time spent): W4A4 is slower, not faster — do not ship

The whole point of this project was "does native FP4 tensor-core compute beat
Marlin's dequant fallback." Measured properly (5 interleaved invocations per
arm, warmup discarded, median + range, `scripts/bench/bench_ab.sh` +
`bench_ab_analyze.py`), the answer is no, on this card, for this workload:

| | median decode | latency range (5 reps) | kernel |
|---|---:|---|---|
| A16 (published) | **18.8 tok/s** | 13.0–15.1 s | `MARLIN` |
| W4A4 (this build) | **14.5 tok/s** | 16.7–17.9 s | `FLASHINFER_CUTLASS` |

W4A4 runs at **0.77x** A16's speed, batch=1, in=512/out=256, **eager mode**.
The ranges don't overlap — this isn't noise, but it also isn't the whole
story: `bench_ab.sh` deliberately runs `--enforce-eager` to isolate kernel
dispatch from CUDA-graph effects, and this repo's own README documents that
CUDA graphs are worth ~7x on this card for A16 (eager 19.9 → PIECEWISE 139.4
→ FULL_AND_PIECEWISE 149.5 tok/s). The eager-only number above was correctly
flagged (by the user, not caught internally first — worth being honest about
that) as an incomplete basis for a publish decision, since W4A4 had never
been tried under CUDA graphs at all and a documented third-party bug class
(`lna-lab` patch #10: PyTorch Inductor bugs specifically in piecewise CUDA
graphs + NVFP4 activation-quant fusion) meant it wasn't obviously safe to
assume it would even run, let alone what it would measure.

**CONFIRMED under PIECEWISE CUDA graphs (2026-08-22, the production-representative
comparison)** — smoke-tested clean first (2/2 healthy, no Inductor crash;
the patch #10 risk class did not materialize on this stack), then measured
with the same 5-rep/interleaved/median+range methodology,
`gpu_memory_utilization=0.92` + `--kv-cache-dtype fp8` (needed for A16 to
even allocate KV cache under PIECEWISE at this box's current VRAM headroom —
see the memory-margin finding below, which applies here too):

| | median decode | range (5 reps) |
|---|---:|---|
| A16 (Marlin) | **142.5 tok/s** | 126.5–146.6 |
| W4A4 (native FP4) | **119.2 tok/s** | 114.3–128.7 |

A16's number here (142.5) lines up with the previously-published PIECEWISE
figure (139.4) — good cross-validation the methodology is sound. **W4A4 is
0.84x A16 under PIECEWISE — the direction from the eager-mode result holds.**
Interesting wrinkle: W4A4 gains proportionally *more* from CUDA graphs than
A16 does (eager→PIECEWISE: A16 7.6x, W4A4 8.2x), so the gap narrows from
0.77x to 0.84x, but it does not close or reverse. A16 remains faster in
absolute terms under both eager and graph-captured execution.

Combined with the accuracy suite (mixed, not better: HumanEval 92.07% vs
A16's 95.7%, HumanEval+ 89.02% vs 90.9%, MBPP+ 91.01% vs 89.9%), there is no
remaining case for this checkpoint: it is both slower and not more accurate
than what's already published, **now checked under the actual production
serving mode, not just eager.** **Do not replace the A16 release with this,
and do not publish it alongside it.** The theoretical case for W4A4 (native
FP4×FP4 tensor-core instructions should beat a dequant-then-bf16-compute
kernel) does not survive contact with measurement on SM120 consumer
Blackwell for a single-stream decode workload, under either eager or
graph-captured execution — worth writing up honestly as a negative result,
since the earlier optimism (see the "Next major project" framing below, and
the retracted "~+31% throughput" claim) was wrong and the reproducible
reason is now on record. Raw JSON/logs: `~/bench-ab-piecewise/`.

**A real, separate finding surfaced getting this measurement**: at identical
`vllm bench latency` settings (`gpu_memory_utilization=0.90`,
`max_model_len=2048`), A16/Marlin failed to allocate KV cache on **every**
invocation (6/6, both the unstripped and the properly-stripped checkpoint),
while W4A4 succeeded on every invocation at the same settings. Marlin's
dequant-to-bf16 path needs meaningfully more non-weight runtime workspace
than the native FP4 kernel — the checkpoint sizes are nearly identical, so
this isn't a size effect. A16 only became bench-able at
`gpu_memory_utilization=0.91` + `max_model_len=1024` (still hit the same
razor-thin-margin failure on 1 of 6 attempts even then — Windows-desktop
VRAM contention on this box, already documented elsewhere in this repo, is
close enough to the edge that this specific benchmark shape is fragile).
This has no bearing on the production serving config (`kat_overrides_sota.yaml`
already runs A16 successfully at 0.92 with different settings) — it's
specific to `vllm bench latency`'s memory-profiling path — but if this
benchmark is ever re-run, start there rather than rediscovering it.

Detail, raw JSON, and logs: `~/bench-ab/` on the WSL box (not yet committed
to the repo — do that if this negative result is written up anywhere
public).

---

Scope for that project (steps 1-4 done 2026-08-21, see RESULT above; 5-7 were
audit/prep, not blocked on the negative result):

1. ~~Run `scripts/quantize/quantize_kat_w4a4.py`~~ — done. 4096-token
   calibration, 29.7 min, verified `weights=4 acts=4`, 128 experts, stripped
   to 12.4532 GiB.
2. ~~Smoke-test~~ — done, with a real finding: the FIRST invocation crashed
   (`CUDA error: an illegal memory access`) during the very first post-JIT
   warm-up forward pass. A rerun with `CUDA_LAUNCH_BLOCKING=1`, reusing the
   same compiled kernels, passed clean (`SMOKE_PASS`, 4/4 coherent). Root
   cause was never isolated (no Xid/dmesg signal available in WSL2) — treat
   as a one-time JIT-compile-adjacent hiccup, not a confirmed-safe kernel,
   if this checkpoint is ever touched again after a `~/.cache/flashinfer`
   clear.
3. ~~Re-run the accuracy suite~~ — done, see RESULT above.
4. ~~Measure actual throughput~~ — done, see RESULT above. This is also the
   correction of the retracted "~+31% throughput, already measured" claim
   that was in this file earlier the same day.
5. ~~Revisit the `lna-lab/blackwell-geforce-nvfp4-gemm` patch set~~ — done
   2026-08-21 without spending GPU/CPU time: all four patches critical to MoE
   FP4 on SM120 are already upstream in this box's installed
   flashinfer/vllm-src. Nothing to apply.
6. `VLLM_USE_AOT_COMPILE=1` JIT-cost test — moot given the result above; not
   worth pursuing for a checkpoint that isn't shipping.
7. **Publish strategy: don't.** Keep A16 as the sole released checkpoint.
   Leave this session's W4A4 build and benchmark data on disk
   (`~/models/kat-50pct-nvfp4-w4a4-stripped`, `~/bench-ab/`) as a documented
   negative result rather than deleting it — the finding itself (native FP4
   loses to Marlin+dequant on SM120 consumer Blackwell for single-stream
   decode) is worth keeping and citing if this gets revisited, e.g. at
   higher batch sizes where the native kernel's throughput-over-latency
   tradeoff may look different (not measured — this session only tested
   batch=1, matching the agentic single-stream use case the model targets).

The `~/vllm-env-027` / `~/vllm-src-027` build (vLLM 0.27.1, SM120, torch
2.13.0+cu130) is left on disk as a starting point if this project wants to
revisit the two SM120-specific decode-throughput fixes noted in
`docs/optimization_research_2026-08-21.md` §1 — those didn't help the A16
checkpoint but weren't tested against W4A4's different kernel path.

## Longer-horizon / not scheduled

- **Candidate performance levers surfaced 2026-08-22, not yet tested against
  this checkpoint — side note for the next experiment, not acted on this
  session:**
  - [FreeToken](https://github.com/FlashML-org/FreeToken) (FlashML-org, 503
    stars, paper arXiv:2608.16157, authors include Song Han/Ion
    Stoica/Matei Zaharia) — an edge-native MoE serving engine (not a vLLM
    fork; a different stack), pitched specifically at running frontier MoE
    models on consumer hardware via bandwidth-adaptive CPU-GPU co-execution
    and a global LRU **expert cache** with elastic VRAM reallocation between
    expert cache and KV cache, no restart needed. Native NVFP4 support,
    explicit RTX 50-series support. Why it matters: this project's entire
    premise has been "prune 50% of experts to fit 16 GB" — if expert
    offloading genuinely works, it could let this project serve the
    **unpruned or much-less-pruned** base model instead, skipping the REAP
    accuracy tax entirely, a different and potentially bigger lever than
    anything pulled this session. Unverified at this project's scale;
    brand-new repo (created 2026-07-20). Would need real install +
    compatibility testing against the pruned checkpoint before trusting it
    over vLLM.
  - [Qwen-Sharp-Chat-Templates](https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates)
    (172 likes) — a drop-in `chat_template.jinja` for Qwen3.5/3.6/3.8 (this
    model's base family) that fixes the exact `<think>`-tax issue this
    project already documented itself (`kat_overrides_sota.yaml`'s own
    comment: "the `<think>` preamble alone consumed 300 tokens without
    finishing"). Author-claimed, not independently verified: -59% answer
    tokens on a Claw-Eval benchmark, 2.5x faster median time-to-fix on
    SWE-bench-Live at equal resolve rate. Also fixes tool-call-escalation
    bugs (long tracebacks not triggering retry, false-positive retry loops
    on code search) — same bug *class* this project had to patch itself for
    SWE-bench (`--enable-auto-tool-choice`, the litellm tool-calling fix in
    `CHANGELOG.md`). Much cheaper to test than FreeToken — a one-file
    template swap, no re-quantization or serving-stack change — but tuned
    against upstream Qwen3.5/3.6/3.8, not against KAT-Coder's specific
    REAP-pruned, MTP-field-carrying fork, so verify it doesn't break this
    model's chat format/tool-call parsing before trusting the claimed gains.
    If this pans out, it's orthogonal to the W4A4/A16 quantization-scheme
    question entirely — a token-efficiency lever, not a kernel-throughput
    one, and could stack with whichever checkpoint gets used.

- GGUF quant for reach (llama.cpp/Ollama/LM Studio compatible) — current
  model has 74 HF downloads (verified 2026-08-21); NVFP4A16 needs vLLM +
  Blackwell specifically, which caps the addressable audience. Re-verified
  2026-08-21, correcting an error introduced earlier in this same session
  (see `docs/optimization_research_2026-08-21.md` addendum for the full
  story): the same author (`gbuzhf`) publishes two distinct GGUF repos for
  this base model —
  [`KAT-Coder-V2.5-Dev-MTP-GGUF`](https://huggingface.co/gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF)
  (45.3K downloads, full unpruned model, MTP head grafted from
  Qwen3.6-35B-A3B) and
  [`KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF`](https://huggingface.co/gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF)
  (5.3K downloads, REAP-pruned **256→205 experts, 19.9%** — a much lighter
  prune than our 50% — same grafted MTP head, KLD-measured against the
  unpruned original: mean KLD 0.059, 94.6% top-1 token agreement, no
  downstream coding benchmark run). `README.md`'s "Honest positioning" and
  `HF_MODEL_CARD.md`'s "Prior art" sections cite the second (205E) repo
  correctly — no fix needed there. If we ship a GGUF, the useful comparison
  is prune depth (50% vs. their 19.9%) and that we'd publish SWE-bench
  numbers where they explicitly do not.
- Base-model swap to Ornith-1.5-35B-A3B — tracked separately in the
  `sota-ornith-build` branch of the private 60%-experiment repo, not this one.
