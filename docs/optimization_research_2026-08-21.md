# SOTA audit + optimization research — 2026-08-21

Follow-up to `optimization_research_2026-08-20.md`'s "Serving: vLLM 0.26.0 →
UPGRADE" row and "Spec decode: DFlash when vLLM PR merges" row. Both were tested
against this checkpoint tonight. Verdict for both: **do not adopt**, with
evidence below. No config changed as a result of this session; the existing
`kat_overrides_sota.yaml` / `run_pilot_all.sh` configuration (fp8 KV cache,
`MAXLEN=49152 MAXSEQS=2`) remains correct and untouched.

## 0. Verdict summary

| Component | Tested | Verdict | Why |
|---|---|---|---|
| vLLM 0.27.1 (source build, SM120, torch 2.13.0+cu130) | Built and smoke-tested | **REJECT** | No kernel-path change for our checkpoint; ~1 GiB more fixed memory overhead than 0.26.0, enough to make KV cache allocation fail at settings that work fine on 0.26.0 |
| DFlash speculative decoding | Researched, not built | **N/A** | Real technique (vLLM ≥0.20.1, 6x+ lossless), but requires a trained drafter for the target model family. None exists for KAT-Coder / this Qwen3.5-hybrid architecture; training one is a separate project |
| Native MTP | Checked checkpoint config | **N/A** | `mtp_num_hidden_layers: 0` in our checkpoint — the base model supports the field but ships no MTP weights. Competitor GGUF (`gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF`) only has MTP because they grafted a head from a sibling model (Qwen3.6-35B-A3B); not a same-project change |
| `lna-lab/blackwell-geforce-nvfp4-gemm` community SM120 patches | Read patch source, diffed against our vLLM checkout | **REJECT (for A16), REVISIT (for W4A4)** | The patches fix a device-family gate (`is_device_capability_family(120)`) already merged upstream in 0.27.1. Our real MoE-kernel rejection is scheme-level, not device-level — see §2. Patches may still matter once a W4A4 checkpoint exists (see roadmap) |
| NVFP4 KV cache (`--kv-cache-dtype nvfp4`) | Started server on 0.26.0 | **REJECT** | `nvfp4` is a declared `CacheDType` option but no attention backend (FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT) implements it for this model's config (`head_size=256`, hybrid mamba/attention). Hard `ValueError: No valid attention backend found`, not a tuning problem |
| CPU parallelism / architecture scoping for the 0.27.1 build itself | Tuned during build | Neutral (build-process finding only) | CMake already auto-scopes `CUDA target architectures` to the host GPU (12.0 only) with no `TORCH_CUDA_ARCH_LIST` override needed — this is not model-serving-relevant, recorded here only so a future rebuild doesn't waste time re-deriving it |

## 1. vLLM 0.27.1 — built, smoke-tested, rejected

Built from source (`~/vllm-src-027`, tag `v0.27.1`) into a separate venv
(`~/vllm-env-027`) so the working `~/vllm-env` (0.26.0) was never at risk.
Torch 2.13.0+cu130 (already proven elsewhere on this box, in `~/quant-env`),
FlashInfer 0.6.16.post3, MAX_JOBS=12 (this card's 12-core WSL allocation).
Both artifacts are left on disk (~17 GiB) as a reference/fallback rather than
deleted — see roadmap.

**Kernel path, unchanged.** Loading the checkpoint still logs:

```
WARNING [marlin_utils_fp4.py:354] Your GPU does not have native support for
FP4 computation but FP4 quantization is being used. Weight-only FP4
compression will be used leveraging the Marlin kernel.
```

— the identical warning 0.26.0 produces. `vLLM`'s own MoE-backend oracle
confirms why (see §2): the rejection is scheme-level, and no vLLM version can
fix that for an activation-unquantized checkpoint.

**New finding: memory regression.** At `gpu_memory_utilization=0.90` (the
value 0.26.0 runs comfortably on, per `smoke_pruned_nvfp4.py`'s own comment:
"leaving ~1 GiB over the weights, which is plenty"), 0.27.1 fails:

```
INFO [gpu_worker.py:563] Available KV cache memory: -0.3 GiB
ValueError: No available memory for the cache blocks.
```

Free VRAM at the time of the test (14.66 / 15.92 GiB, ~1.26 GiB held by the
Windows desktop) exactly matches the historical number in
`smoke_pruned_nvfp4.py`'s own comments — so this is not new Windows-side
contention, it is 0.27.1 itself reserving more. Raising
`gpu_memory_utilization` cannot close the gap: 0.92 leaves only 0.04 GiB spare
(sometimes -0.1 GiB — normal run-to-run Windows VRAM variance of a few hundred
MiB tips it over), and 0.94 exceeds a new hard startup gate 0.27.1 added
(`Free memory on device ... is less than desired GPU memory utilization`).
Confirmed the shortfall is not context-size-dependent: `max_model_len=512`
fails identically to `max_model_len=2048`, so this is fixed overhead, not
something that scales away with a smaller run. Root cause not isolated
(candidates: new per-engine profiling mechanics, additional kernel workspace
reservations, PyTorch 2.13 baseline overhead) — not worth further time against
zero proven upside (next point).

**No proven speed upside either.** The two 0.27.1 commits that plausibly help
this exact card (`92c7fac6` NVFP4 scale zero-init for Blackwell decode
throughput, `9a08a511` skip cooperative top-K on SM120) touch MoE
routing/decode paths, not the Marlin GEMM kernel that dominates. We never
measured an actual token, since the negative-KV-cache crash happens before
generation — so this is a documented absence of evidence, not a documented
absence of benefit. It just wasn't worth chasing given §2 below independently
rules out a MoE-kernel win regardless of vLLM version.

## 2. Why MoE experts fall back to Marlin — scheme, not device

This is the most important finding of the night, because it rules out an
entire category of future fixes (any vLLM upgrade, any device-capability
patch) for the currently-shipped NVFP4A16 checkpoint.

vLLM's `NvFp4` MoE backend oracle logs its full decision on every load
(`VLLM_LOGGING_LEVEL=DEBUG`):

```
NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support ... kernel does not
    support current device cuda.
NvFp4 MoE backend 'FLASHINFER_CUTEDSL' does not support ... same reason.
NvFp4 MoE backend 'FLASHINFER_CUTEDSL_BATCHED' does not support ... same reason.
NvFp4 MoE backend 'FLASHINFER_CUTLASS' does not support ... kernel does not
    support quantization scheme QuantKey(u8, scale(f8e4m3fn, static,
    GroupShape(row=1, col=16)), scale2(f32, static, per_tensor), symmetric)xNone.
NvFp4 MoE backend 'VLLM_CUTLASS' does not support ... same scheme reason.
Using 'MARLIN' NvFp4 MoE backend out of potential backends: [...]
```

Read directly from `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`
(0.27.1 checkout): `CutlassExpertsFp4._supports_current_device()` **already**
includes `p.is_device_capability_family(120)`, and
`flashinfer_cutlass_moe.py`'s `FlashInferExperts._supports_current_device()`
**already** includes it too. These are exactly what
`lna-lab/blackwell-geforce-nvfp4-gemm`'s patches #3 and #4 ("Critical
priority") add — both already upstream. Applying them would change nothing:
the rejection above isn't a device-family check failing, it's a
`QuantKey` scheme mismatch.

The mechanism: real FP4 tensor-core MMA instructions consume FP4×FP4 operands.
Our checkpoint is NVFP4A16 — 4-bit weights, activations left at bf16 — so there
is no matching hardware instruction for the CUTLASS/FlashInfer kernels to
target, on *any* SM family, datacenter Blackwell included. Marlin's
dequant-then-compute fallback is the only kernel that can execute a weight-only
scheme at all. This generalizes and confirms, at the MoE-expert level, what
the 2026-08-20 entry already measured at the per-linear-layer level (W4A4
reaches `FlashInferCutlassNvFp4LinearKernel`, A16 cannot).

The community patch repo's own reference recipe quantizes with `scheme: NVFP4`
(W4A4, not NVFP4A16) — consistent with this: their benchmarks work because
their checkpoints produce a `QuantKey` the native kernels actually support.

## 3. NVFP4 KV cache — declared, not implemented for this model

```
ValueError: No valid attention backend found for cuda with
AttentionSelectorConfig(head_size=256, ..., kv_cache_dtype=nvfp4, ...).
Reasons: {FLASH_ATTN: [kv_cache_dtype not supported], FLASHINFER: [...],
TRITON_ATTN: [...], FLEX_ATTENTION: [...], TURBOQUANT: [...]}.
```

`nvfp4` is present in `vllm/config/cache.py`'s `CacheDType` literal on 0.26.0,
but zero attention backends implement it for a `head_size=256` hybrid
mamba/attention config. Declared-but-unimplemented, not a tuning gap — fp8 KV
cache (current config) stays correct.

## Next

See `ROADMAP.md`. Nothing in tonight's findings changes the plan to run the
50-instance SWE-bench pilot at `MAXLEN=49152 MAXSEQS=2` — that number was
already measured and validated on 2026-08-21 before this investigation started,
and remains the target for the next session.

## Addendum: a self-correction, and a correction of that correction

A later self-audit pass this same session checked README.md's "Honest
positioning" claim that `gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF` exists
as a REAP-pruned GGUF of this base model. The check was done by web-searching
and fetching the closest-named result, `gbuzhf/KAT-Coder-V2.5-Dev-MTP-GGUF`
(no "REAP-205E" segment) — which turned out to be a *different, real* repo by
the same author: the full unpruned model with a grafted MTP head, no
pruning at all. Concluding from that mismatch that README's claim was wrong,
`ROADMAP.md` and a prior-session memory note were both "corrected" to say the
205-expert REAP GGUF didn't exist.

It does. Fetched directly by its exact repo ID (`gbuzhf/KAT-Coder-V2.5-Dev-REAP-205E-MTP-GGUF`),
not a search result: real repo, REAP-pruned 256→205 experts (19.9%,
data-free saliency ranking), same grafted MTP head, 5.3K downloads, with its
own KLD-based quality disclosure (mean KLD 0.059, 94.6% top-1 agreement, no
coding benchmark run). README.md and HF_MODEL_CARD.md's original claims were
correct and needed no fix. `ROADMAP.md` and memory have been corrected back,
this time citing both of the author's two distinct repos by exact ID.

**Why this happened**: a fuzzy match (web search for a plausible name) was
treated as equivalent to checking the exact claimed identifier. The lesson
generalizes beyond this one fact: verifying a specific claim means looking up
that specific identifier, not a similar one that a search surfaces first.
