# SOTA audit + optimization research — 2026-08-20

Audit question: is our build (REAP-pruned + NVFP4A16 KAT-Coder-V2.5-Dev, vLLM on RTX
5070 Ti / SM120, 16 GB VRAM) the state of the art this hardware can run for
agentic SWE-bench? Findings below, each with a verdict and a source. Folding
mechanism: `rebuild/05_optimize.sh` measures each APPLY item as a GATE and adopts
it only on a measured win (ties keep stock). Verdicts land in
`~/kat_swebench/runtime_config.env`, sourced by `04_swebench_test.sh` and the
serve scripts (`KAT_TEMPLATE`, `KAT_ATTN_VERSION`, `KAT_SPEC_CONFIG`).

## 0. Verdict summary

| Component | Our build | SOTA for 16 GB SM120 | Action |
|---|---|---|---|
| Base model | KAT-Coder-V2.5-Dev (69.4 SWE-bench claimed) | **Ornith-1.5-35B-A3B (79.0 claimed, released 2026-08-19)** | **SWITCH (A/B it)** |
| Pruning | REAP 50% (256->128 experts) | REAP 50% **with code-domain calibration** | KEEP, add router-protect |
| Quantization | PTQ NVFP4A16 | **GPTQ NVFP4A16** (same size, better recovery) | SWITCH |
| Serving | vLLM 0.20.2 | **vLLM 0.26.0** (hybrid prefix-cache fixes) | UPGRADE |
| KV cache | fp8 e4m3 | fp8 e4m3 (never --calculate-kv-scales) | KEEP (correct) |
| Attention | FA4 default (SM120) | FA2 vs FA4 = gate A (unresolved) | MEASURE (GATE A) |
| Spec decode | MTP planned | **DFlash when vLLM PR merges** (biggest win) | GATE D (runtime check) |
| Template | Sharp v22.1 + noretain | **Sharp rebased to v22.2** (grep fix) | DONE (this commit) |

## 1. Qwen-Sharp-Chat-Templates — APPLY (highest leverage per dollar)

**Status: FOLDED as GATE T, rebased to v22.2 in this commit.**

Froggeric's Qwen-Fixed-Chat-Templates + a terseness system prompt. Drop-in
`chat_template.jinja` for Qwen3.5/3.6/3.8 — our model is Qwen3.5-family.

- **v22.2 (2026-08-19) is the current upstream**; the Sharp repo is still v22.1.
  v22.2 fixes a bug that matters to a SWE agent: tool outputs containing
  `throw new Error` / `console.error` / `logger.error` / `def ` / `function ` /
  `import ` in the first 80 chars were misread as failures, injecting fake
  "SYSTEM WARNING" retry loops. We vendored v22.2 and re-applied the 11-line
  Sharp terseness splice ourselves (upstreams may rebase; we no longer depend on
  the Sharp repo). Also adds effort aliases (`ultracode`/`extreme`/`max`) and
  multi-system-prompt merging.
- Community evidence for our exact case: `filipmihal/swe-bench-qwen` ran
  mini-swe-agent + Qwen3.6-35B-A3B (the SAME 256-expert 35B/3B MoE family as
  KAT) at **57.0% SWE-bench Verified (285/500)** with
  `T=1.0, top_p=0.95, preserve_thinking=true`. That run used the STOCK Qwen
  template; Sharp is strictly terser, so this is a floor, not a ceiling.
- Author-measured (same weights, 27B): +7.4 answer score / -59% answer tokens on
  Claw-Eval; MMLU-Pro tokens-per-correct -22%; SWE-bench-Live "~2x faster to a
  fix". MoE caveat: their 35B/3B MoE (Nail) spends 5.8k thinking tokens/question
  on GPQA vs 2.4k for a dense 27B — terseness matters MORE for our MoE shape.
- Knob reference (v22.2): `reasoning_effort` (none|minimal|low|medium|high|xhigh|
  ultracode|extreme|max; default medium = zero tokens injected), `preserve_thinking`
  / `preserve_reasoning` (default true; retention = prefix-cache friendly but
  costs 60-120k tokens over a 30-turn session), `tool_call_format` (xml|json),
  `auto_disable_thinking_with_tools`, `max_tool_arg_chars`, `max_tool_response_chars`,
  inline `<|think_*|>` tags. `chat_template_kwargs` is the only channel that
  works for effort steering (OpenAI top-level `reasoning_effort` is consumed by
  the server and never reaches the template).
- Our recommendation: **noretain variant for 32K** (retention grows the prompt
  past the ceiling by ~turn 10-20; the template still keeps the current turn's
  thinking via `last_query_index`). If we later relax to 128K (Ornith base,
  see §7), switch to retain and let prefix caching win.
- Licensing: Apache-2.0 end-to-end (template + Qwen bases). Safe to ship.

## 2. Ornith-1.5 — SWITCH BASE (released 2026-08-19)

**Corrected verdict vs previous WATCH.** Ornith-1.5-35B-A3B is byte-for-byte the
same architecture as KAT-Coder-V2.5-Dev: same `qwen3_5_moe` arch, 256 experts,
8/256 active, 40 layers with the identical 30-linear+10-full-attention layout,
same vocab 248320, same vision stub. Differences that matter: it has
`mtp_num_hidden_layers: 1` (MTP weights present — GATE M becomes viable) and it
claims SWE-bench Verified **79.0** (5-run avg, OpenHands, temp 1.0, anti-hack
safeguards) vs KAT's claimed 69.4. Our entire pipeline (REAP prune + NVFP4
quantize + serve) is a drop-in for it.

Caution: these are cross-harness claims. Same-harness deltas are ~10pp (KAT's
own eval of Qwen3.6-35B-A3B = 64.4 vs official 73.4). Expect a smaller but real
head start. ACTION: run the 60%-experiment on KAT for the ratio data, then
re-run the SOTA build on Ornith-1.5-35B-A3B at 50% + GPTQ-NVFP4A16 + v22.2
template. The pipeline scripts are BASE_MODEL-parameterized for exactly this
swap (env `BASE_MODEL`, `BASE_DIR`, `QUANT_SRC`/`QUANT_DST`).

## 3. LFM2.5-2.6B — WATCH, architecture reference only (unchanged)

2.6B hybrid (22 double-gated short-conv blocks + 8 GQA), 128K context in 2.5 GB,
agentic RL inside real harnesses. Explicitly "not recommended for agentic
coding" — cannot replace KAT. Transferable lessons: (a) hybrid linear attention
= long context in small memory (KAT already has this: only 10/40 layers hold KV);
(b) agentic RL in-harness is the frontier recipe we can only approximate via
harness-level config. LFM2.5 needs vLLM >= 0.23.0 (fine after our 0.26.0
upgrade). Not a base-model candidate.

## 4. "D Flash" — IDENTIFIED: speculative decoding, not an attention kernel

**Corrected verdict vs previous section.** "DFlash" is **DFlash: Block Diffusion
for Flash Speculative Decoding** (Z-Lab, MIT Han Lab; arXiv 2602.06036, ICML
2026) — a speculative-decoding framework whose *draft model is a block diffusion
model* that drafts whole token blocks in one forward pass; the target verifies in
parallel (lossless). Claims >6x speedup, up to 2.5x over EAGLE-3. Hardware-agnostic
(consumer GPU proven on a single RTX 4090); stacks with quantized weights; has
drafters for the Qwen3.5 family including 35B-A3B MoE.

- **vLLM status: PR #52816 is OPEN, NOT MERGED (verified via GitHub API
  2026-08-20).** SGLang merged it (PR #35371, LMSYS). llama.cpp + MLX also have
  it. Since we are on vLLM, DFlash is WATCH until the PR lands; GATE D in
  05_optimize.sh checks the installed vLLM at runtime and only then attempts an
  A/B. When available, this is the single biggest agentic-decode win on the
  table (agentic loops are decode-bound).
- Not to be confused with FlashMLA (DeepSeek, SM90/SM100 only, MLA-specific —
  N/A for our GQA+linear arch) or TurboAttention (Microsoft, no public code, N/A)
  or FlashAttention 4 (below).

## 5. Attention kernels on SM120 (RTX 5070 Ti)

- **FlashAttention-4**: upstream SM120 support since 2026-03 (PR #2329/#2330).
  In vLLM since v0.17.0 (2026-03-07), but the authors' own benchmark shows FA4 in
  vLLM wins only on PREFILL and spec-decode; **FlashInfer still wins DECODE**
  (the shape that dominates our agentic loop). Community data: FA4 is 4-10%
  slower than FA2 on short Qwen3 prefill (our 590-token steps). GATE A measures
  FA2 vs FA4 on our exact shapes; keep the winner.
- **FlashInfer b12x**: the CuTeDSL FP4/MXFP4 GEMM + fused-MoE path for SM120
  (since FlashInfer 0.4.0). Accelerates our NVFP4 weights specifically; vLLM
  exposes it as the FLASHINFER_B12X NVFP4 MoE backend. Needs
  `flashinfer-python[cu13]` + JIT. WATCH — measure after FA/MTP; potential
  decode-GEMM win for the NVFP4 compute path.
- CUTLASS/cuDNN/Marlin/BitBLAS: present on SM120 but unbenchmarked for this
  class; no evidence-based reason to switch.

## 6. KV cache quantization — CONFIRMED CORRECT (no change)

- **fp8 e4m3 is the SOTA KV dtype for this model** — near-lossless on Qwen3.5
  (<=0.7pt, >=99% recovery), 54% decode ITL slope improvement on Blackwell.
  Our `--kv-cache-dtype fp8` is right. Never add `--calculate-kv-scales` on the
  hybrid (vLLM #37554: silent corruption; fixed in #37565) — added a warning to
  serve_kat.sh.
- NVFP4 KV on SM120 is NOT ready (SM100-gated until PR #46329, and a reviewer
  reported every-2nd-request corruption on hybrid Qwen). Skip.
- Forward watches (only if we later chase >128K): KV Pareto (int2/4/8 mix,
  EACL 2026) and TurboQuant (K3V2 rotation + Lloyd-Max KV, ~31% savings on hybrid
  DeltaNet models; ICLR 2026). Not needed at 32K.

## 7. MTP speculative decoding — CONDITIONAL (base-dependent)

Supported for qwen3_5_moe in vLLM via
`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` (only MTP-1
works for the hybrid; the `qwen3_5_mtp` alias is deprecated). **Caution from
vLLM maintainers: MTP was -22% on the Marlin W4A16 path on SM120** (draft-head /
activation mismatch) — our NVFP4A16 weights use that path, so acceptance must be
measured, not assumed. KAT has `mtp_num_hidden_layers: 0` (GATE M will correctly
no-op); **Ornith-1.5 has 1** — switching bases makes GATE M meaningful. Recipe
notes: disable prefix caching for MTP; reduce `--max-cudagraph-capture-size` if
the mamba-cache assert fires. GATE M in 05_optimize.sh verifies weights, boot-tests
at 32K (no context shrink), then A/Bs with drift check.

## 8. Pruning — REAP KEEPS, with three corrections

- **REAP remains SOTA** for MoE expert pruning (ICLR 2026; near-lossless at 50%
  on Qwen3-Coder-480B and Kimi-K2, 75 -> 73.1 SWE-bench Verified). Official repo
  fixes landed 2026-03: logit renormalization, calibration mix, layerwise
  calibration observer.
- **Calibration data is the critical variable**: C4-calibrated REAP scored 4.6%
  vs AIMER's 36.1% on code (arXiv 2603.18492). We calibrate on evol-codealpaca —
  correct, keep it. Never C4.
- **"Half the Experts, All the Code" (arXiv 2607.16721)**: pruning criteria do
  NOT transfer across model families (Qwen3.6 vs Gemma-4 flip), fine-tuning
  recovers ~half of pruning loss, and single-shot benchmarks overstate agentic
  ability. Two consequences for us: (a) our layerwise REAP + code calibration is
  family-validated (KAT/Ornith are both Qwen3.5-MoE), (b) **aggressive pruning
  hurts long-horizon editing more than short benchmarks suggest — support for
  staying at 50% rather than 60%.**
- **Pruning beats quantization only below ~3 bits/weight** — at NVFP4 (~2
  bits/weight effective) we are exactly in the regime where pruning is
  beneficial. 50% + NVFP4 is the validated sweet spot; 60% is unvalidated for
  this arch and risks the dominant failure mode (long-horizon editing) we
  already saw at 50% (18/50 CWE).
- **Verdict: SOTA = 50%.** The 60% experiment still runs for data; the shipped
  build goes back to 50% + GPTQ-NVFP4A16 + Ornith base.

## 9. Quantization — SWITCH PTQ -> GPTQ-NVFP4A16

- llm-compressor 0.13.0 is current. **GPTQ FP4 microscale (NVFP4) since 0.10.0
  (2026-03-02) has improved recovery vs plain PTQ.** Same file size (weight-only),
  better accuracy. Switch the recipe from QuantizationModifier to GPTQModifier
  (env `QUANT_MODE=gptq`, default).
- W4A4 is measurably worse than W4A16 (KLD 2-4x worse >10K ctx; NVIDIA forum
  2026-05) — NVFP4A16 stays.
- Mixed-precision protection (ModelOpt NVFP4_EXPERTS_ONLY + community KAT quant
  `sakamakismile/KAT-Coder-V2.5-Dev-NVFP4`): protect lm_head, routers, shared
  expert gates, **DeltaNet conv1d + linear-attention projections**, embeddings,
  MTP. Our ignore list covers all of these; add `re:.*router.*` (the expert
  router weights are small but accuracy-critical). Ornith-style (keep attention
  + dense in FP8/BF16) is a WATCH probe, not a default (size risk on 16 GB).

## 10. Serving stack — UPGRADE vLLM to 0.26.0

- Latest stable is v0.27.1 (2026-08-11). **v0.26.0 (2026-07-27) is the sweet
  spot**: hybrid prefix-cache fixes (#46384, #47782), Qwen3.5 kernel fusions,
  pairs with transformers 5.13. v0.27.x requires torch 2.13 (breaking for our
  venv). Our 0.20.2 pin is functional but 7 releases behind and lacks the hybrid
  prefix-cache fixes. 02_build_vllm.sh takes `VLLM_TAG` (default v0.26.0).
- Qwen3.5 official recipe matches our config except one flag:
  **`--tool-call-parser qwen3_coder` (not qwen3_xml)** — changed in
  run_pilot_all.sh. Everything else (chunked prefill, fp8_e4m3, prefix caching,
  PIECEWISE graphs, max-num-batched-tokens 4096) matches.
- SGLang ruled out for this class: wins dense-model throughput on SM120 (+47%)
  but is broken for hybrid Qwen (NaN outputs, fp8-KV corruption, FP4-KV requires
  disabling radix cache). vLLM's Marlin + FlashInfer-attention + fp8-KV is the
  only demonstrated-working SM120 hybrid path.

## Experiment queue (in priority order)

1. 60% REAP experiment on KAT (this branch) for the ratio data.
2. SOTA build: Ornith-1.5-35B-A3B + 50% + GPTQ-NVFP4A16 + v22.2 Sharp template
   + vLLM 0.26.0 + qwen3_coder parser -> pilot-5 vs the 50% KAT baseline.
3. GATE A: FA2 vs FA4 (bench_ab.sh, 5 interleaved reps).
4. GATE M: MTP (viable after the Ornith switch; measure acceptance on NVFP4 path).
5. GATE D: DFlash when vLLM PR #52816 merges (runtime-checked).
6. FlashInfer b12x FP4 GEMM probe (decode GEMM; after FA/MTP).
7. reasoning_effort sweep (low/medium/high) + adaptive retry (xhigh).
8. KV Pareto / TurboQuant only if chasing >128K context.

## Sources

- https://arxiv.org/abs/2602.06036 (DFlash) / https://github.com/z-lab/dflash
- https://github.com/vllm-project/vllm/pull/52816 (open, verified 2026-08-20)
- https://github.com/vllm-project/vllm/releases (v0.26.0 2026-07-27, v0.27.1 2026-08-11)
- https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates (v22.2, 2026-08-19)
- https://huggingface.co/peculiar-ragdoll/Qwen-Sharp-Chat-Templates
- https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B (config.json verified)
- https://huggingface.co/LiquidAI/LFM2.5-2.6B
- arXiv 2510.13999 (REAP, ICLR 2026) / 2603.18492 (AIMER) / 2606.15716 (MAN/MSAN)
- arXiv 2607.16721 (Half the Experts, All the Code) / 2603.05451 (FA4) / 2604.15804 (Qwen3.5 hybrid)
- https://huggingface.co/filipmihal/swe-bench-qwen (57.0% SWE-bench, mini-swe-agent + Qwen3.6-35B-A3B)
- NVIDIA forums + llm-compressor changelog (GPTQ FP4 0.10.0, W4A16 vs W4A4 KLD)