# The 0.83 GiB phantom vision tower

**Date:** 2026-08-20
**Status:** root cause identified, fixed, verified
**Artifact affected:** `kat-50pct-nvfp4a16-renorm-stripped` (the published release candidate)

## Summary

The release candidate was **13.28 GiB** on disk while the README and the Hugging Face
model card both documented **12.45 GiB**. The documented figure was correct. The
artifact was carrying **333 randomly-initialised vision-tower tensors totalling
0.8318 GiB** — untrained parameters that the base model does not ship and that no
inference path ever reads.

```
13.2831 GiB  (as built)
-0.8318 GiB  (phantom vision tower, 333 tensors, BF16)
=12.4512 GiB (matches the documented 12.45 GiB)
```

The `-renorm-stripped` suffix was only ever half-true: the build stripped the vision
**declaration** from `config.json` and left the vision **weights** in
`model.safetensors`.

## Symptom

A clean rebuild of the release checkpoint from `master`, using the documented
pipeline, produced a 13.28 GiB artifact instead of the documented 12.45 GiB — a
6.7% overshoot. On a card where the measured KV-cache budget swings between
0.49 and 1.41 GiB, an unexplained 0.83 GiB is not a rounding error, so the gap was
treated as a correctness question rather than a documentation typo.

## What was eliminated first

Each of these was checked and cleared before the real cause was found. They are
recorded because they are the plausible suspects, and knowing they are clean is
worth as much as the eventual answer.

| Suspect | Finding | Verdict |
|---|---|---|
| Wrong branch | Working tree was on `feature/60pct-prune`, whose HEAD swaps the base model and defaults `QUANT_MODE=gptq`. Would have produced a different checkpoint entirely. | Real problem, unrelated to size. Switched to `master`; WIP stashed. |
| Stale prune artifacts | `n64_s42` held a leftover prune at compression-ratio **0.60**, not 0.50. | Real problem, unrelated to size. Re-pruned at 0.50/seed 42. |
| `quantize_kat.py` drift | Exactly one commit in the file's entire history (`90f14b8`); never modified since creation. Ignore list unchanged. | Clean |
| Quantization libraries | `llm-compressor==0.13.0`, `transformers==5.14.1` — match `docs/environment.md` pins exactly. | Clean |
| `compressed-tensors` packing change | 0.18.0 installed. Released **2026-08-08**, nine days *before* this repo's first commit, and nothing has been released since. There is no older version the original build could have used. | Clean |
| `reap` fork drift | HEAD `2954ba3` dated **2026-08-17 19:29:20**, two minutes before this repo's first commit `90f14b8` at **19:31:23**. Zero commits behind its own origin. Unmoved across the project's entire history. | Clean |

Every documented variable reproduced faithfully. The gap was therefore *inside* the
artifact, not in the pipeline configuration — which is what motivated reading the
`safetensors` header directly instead of continuing to audit the toolchain.

## Root cause

The base model declares `Qwen3_5MoeForConditionalGeneration` — a multimodal
architecture — but ships **zero** vision weights. Confirmed directly from its
weight map:

```
BASE model KAT-Coder-V2.5-Dev (model.safetensors.index.json)
  tensors in weight_map : 31,333
  visual tensors        : 0
  prefixes present      : ['language_model']
```

The tower is then created out of nothing, four steps in sequence:

1. **`reap` loads the base via `AutoModelForImageTextToText`.** Transformers finds
   the vision parameters absent from the checkpoint and **randomly initialises**
   them. This is stated plainly in the prune log's load report — all 333 tensors
   listed `MISSING`, followed by:

   > `MISSING: those params were newly initialized because missing from the checkpoint.`

2. **`reap` saves the pruned model**, and the random tower is written out with it.
   The pruned bf16 checkpoint contains 333 `model.visual.*` tensors, 0.8318 GiB.

3. **`quantize_kat.py` protects them.** The ignore list carries `re:visual.*` and
   `re:model.visual.*`, so the quantizer skips them and they survive at full BF16
   rather than being compressed to 4 bits. 0.8318 GiB in, 0.8318 GiB out.

4. **The release step strips the config, not the weights.** Removing
   `vision_config`, `vision_start_token_id` and `vision_end_token_id` from
   `config.json` deletes three JSON keys. It does not touch `model.safetensors`.

The ignore-list entry that protects the tower from quantization is correct on its own
terms — quantizing untrained noise would be pointless. The defect is that nothing
downstream ever removed the tensors, so "protected from compression" quietly became
"shipped at full precision".

### Byte-level evidence

| Stage | Tensors | Visual tensors | Visual bytes | Total |
|---|---:|---:|---:|---:|
| Base `KAT-Coder-V2.5-Dev` | 31,333 | **0** | 0 | 69.3 GB |
| After REAP 50% prune | 16,306 | **333** | 0.8318 GiB | 35.3727 GiB |
| After NVFP4A16 quantization | 47,346 | **333** | 0.8318 GiB | 13.2831 GiB |
| Release candidate (fixed) | 47,013 | **0** | 0 | **12.4512 GiB** |

Tensor counts rise across quantization because each quantized weight is stored as
three tensors (`weight_packed`, `weight_scale`, `weight_global_scale`).

Composition of the 13.28 GiB artifact before the fix:

| Group | Size | Share | Storage |
|---|---:|---:|---|
| experts | 8.4376 GiB | 63.55% | U8 packed + FP8 scales |
| linear_attn | 1.8842 GiB | 14.19% | BF16 (ignored — hybrid linear attention) |
| lm_head | 0.9473 GiB | 7.13% | BF16 (ignored) |
| embed_tokens | 0.9473 GiB | 7.13% | BF16 (ignored) |
| **visual** | **0.8318 GiB** | **6.27%** | **BF16 — untrained noise** |
| other / shared_expert / routers | 0.2287 GiB | 1.73% | mixed |

### Independent corroboration

The smoke-test run confirms the tower was already dead weight at inference time.
vLLM, running with `language_model_only=True`, reported:

```
Checkpoint size: 13.28 GiB
Model loading took 12.58 GiB memory
```

vLLM read a 13.28 GiB file and put 12.58 GiB on the GPU, because it skipped the
phantom tower on the way in. The ~12.45 GiB of real weights plus runtime overhead
was always the true model; the extra 0.83 GiB only ever existed on disk.

## Fix

`scripts/release/build_release_candidate.py` replaces the previous inline
config-only strip. It performs the full release build:

1. copies every sidecar file from the quantized checkpoint
2. removes the vision keys from `config.json`
3. removes the now-dead `model.visual.*` entries from `quantization_config.ignore`
   (110 of them, referring to modules that no longer exist)
4. rewrites `model.safetensors` without any `visual` tensor, by byte-range copy —
   no torch, no dtype round-trip, no full-file load into RAM
5. verifies by artifact: re-parses the written header, asserts zero visual tensors
   remain, and checks the resulting size against the documented 12.45 GiB

## Verification

```
tensors kept    : 47,013
tensors dropped : 333  (0.8318 GiB)
before          : 14,262,571,176 bytes = 13.2831 GiB
after           : 13,369,387,560 bytes = 12.4512 GiB
visual tensors remaining: 0
header re-parsed OK     : 47,013 tensors
OK: 12.4512 GiB matches the documented 12.45 GiB
RELEASE_CANDIDATE_OK
```

## What this changes, and what it does not

**Does:** removes 0.83 GiB (6.27%) of untrained random parameters from every
download; makes the artifact match its own published size; removes 110 stale
entries from the quantization ignore list.

**Does not:** free any VRAM. vLLM's `language_model_only=True` already declined to
load the tower, so resident weights are ~12.58 GiB before and after. This is a
download-size and artifact-correctness fix, not a memory optimisation. Serving
behaviour is unchanged: with no `vision_config` present nothing looks for the
tensors, and any loader that wanted a tower would random-initialise one exactly as
the base model already causes it to.

## Prevention

- The release build now fails (`RELEASE_CANDIDATE_FAIL`, non-zero exit) if any
  `visual` tensor survives, rather than silently shipping them.
- Size is asserted against the documented figure at build time, so a future drift
  between the artifact and the model card surfaces during the build instead of
  after publication.
- General lesson, consistent with this repo's existing "judge by artifacts, never by
  exit codes" rule: a checkpoint's *declared* contents and its *actual* tensor
  inventory are separate things, and only the second one is authoritative. The
  quantizer's own verification block reported `OK: fits a 16 GB card` for the
  13.28 GiB build, because it only checked against a 15 GiB ceiling and never
  against the size the project had published.

## Related defects found in the same pass

- **`scripts/bench/smoke_pruned_nvfp4.py` ignored `KAT_MODEL`.** The model path was
  hardcoded, so the documented step-5 invocation
  (`KAT_MODEL=... ~/vllm-env/bin/python scripts/bench/smoke_pruned_nvfp4.py`)
  silently validated a *different* checkpoint than the one being shipped. Now reads
  the environment variable and defaults to the release candidate.
- **The smoke test's 300-token cap produced a false failure.** KAT emits a `<think>`
  reasoning preamble; on one prompt the entire budget was spent before any code was
  emitted, and the run was scored `no recognisable code`. The cap is now 768
  (`KAT_MAX_TOKENS`), and truncation is reported distinctly from incoherence so a
  harness artifact can no longer masquerade as a model defect.
