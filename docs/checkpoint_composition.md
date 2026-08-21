# Checkpoint composition

What the released checkpoint contains, why it is 12.45 GiB, and why the build
removes a vision tower that the base model never trained.

## The declared tower

`Kwaipilot/KAT-Coder-V2.5-Dev` declares `Qwen3_5MoeForConditionalGeneration`, a
multimodal architecture. It ships no vision weights whatsoever:

```
BASE model KAT-Coder-V2.5-Dev (model.safetensors.index.json)
  tensors in weight_map : 31,333
  visual tensors        : 0
  prefixes present      : ['language_model']
```

Loading a checkpoint like this through `AutoModelForImageTextToText` does not fail.
Transformers constructs the declared modules and **randomly initialises** every
absent parameter, reporting it in the load report:

```
model.visual.blocks.{0...26}.attn.qkv.weight       | MISSING
...
MISSING: those params were newly initialized because missing from the checkpoint.
```

That is 333 tensors of untrained noise, 0.8318 GiB in BF16, attached to a model
that has no use for them.

## Why the tower propagates

Nothing downstream discards those parameters unless told to:

1. **Pruning.** `reap` loads the base model, and saves what it loaded. The pruned
   bf16 checkpoint carries all 333 tensors.
2. **Quantization.** The ignore list contains `re:visual.*` and `re:model.visual.*`.
   This is correct on its own terms — compressing untrained noise would be
   pointless — but "not quantized" means "written out unchanged at full BF16".
3. **Config stripping.** Removing `vision_config`, `vision_start_token_id` and
   `vision_end_token_id` from `config.json` deletes three JSON keys. It does not
   touch `model.safetensors`.

The result is a checkpoint whose *declaration* is text-only and whose *weights*
are not. Config-level stripping alone leaves the full 0.83 GiB in the file.

| Stage | Tensors | Visual tensors | Visual bytes | Total |
|---|---:|---:|---:|---:|
| Base `KAT-Coder-V2.5-Dev` | 31,333 | 0 | 0 | 69.3 GB |
| After REAP 50% prune | 16,306 | 333 | 0.8318 GiB | 35.3727 GiB |
| After NVFP4A16 quantization | 47,346 | 333 | 0.8318 GiB | 13.2831 GiB |
| **Released checkpoint** | **47,013** | **0** | **0** | **12.4512 GiB** |

Tensor counts rise across quantization because each quantized weight is stored as
three tensors: `weight_packed`, `weight_scale`, `weight_global_scale`.

## What the release build does

`scripts/release/build_release_candidate.py` produces the shippable artifact:

1. copies the sidecar files from the quantized checkpoint
2. removes the vision keys from `config.json`
3. removes the dead `model.visual.*` entries from `quantization_config.ignore`
   (110 of them, naming modules that no longer exist)
4. rewrites `model.safetensors` without any `visual` tensor, by byte-range copy —
   no torch, no dtype round-trip, no full-file load into RAM
5. verifies by artifact: re-parses the written header, fails if any visual tensor
   survives, and checks the resulting size against the documented figure

Step 4 is a byte-range copy rather than a load-and-resave because the checkpoint
mixes `U8`, `F8_E4M3`, `BF16` and `F32`; round-tripping those through a tensor
library risks silent dtype coercion for no benefit. Offsets are recomputed
contiguously and the header is re-read after writing to confirm the result parses.

## Composition of the released checkpoint

| Group | Size | Share | Storage |
|---|---:|---:|---|
| experts | 8.4376 GiB | 67.80% | U8 packed 4-bit + FP8 block scales |
| linear_attn | 1.8842 GiB | 15.14% | BF16 — hybrid linear attention, not quantized |
| lm_head | 0.9473 GiB | 7.61% | BF16 — ignored |
| embed_tokens | 0.9473 GiB | 7.61% | BF16 — ignored |
| other | 0.1431 GiB | 1.15% | mixed |
| shared_expert | 0.0661 GiB | 0.53% | U8 packed + FP8 scales |
| routers / gates | 0.0195 GiB | 0.16% | BF16 — ignored |

Experts dominate, which is what makes 50% expert pruning the lever that decides
whether this model fits the card at all. The four ignored groups
(`linear_attn`, `lm_head`, `embed_tokens`, routers) account for 30.5% of the
released weights and are deliberately left at full precision: KAT is 3:1 hybrid
linear attention, and quantizing the linear-attention projections or the routers
costs accuracy for very little size.

## Verification

The 12.45 GiB figure is confirmed independently at three levels.

**Build time.** The release script asserts zero surviving visual tensors and
compares the output against the documented size:

```
tensors kept    : 47,013
tensors dropped : 333  (0.8318 GiB)
after           : 13,369,387,560 bytes = 12.4512 GiB
visual tensors remaining: 0
```

**Load time.** vLLM independently measures the checkpoint and reports
`Checkpoint size: 12.45 GiB`.

**Serving.** `scripts/bench/smoke_pruned_nvfp4.py` passes 4/4 against the released
artifact, with NVFP4 Marlin kernels active and no token collapse.

### Confirming the cause

Before concluding that the tower was the discrepancy, every documented input to
the build was checked against its specification. All reproduced faithfully, which
is what localised the difference to the artifact rather than the pipeline:

| Input | Finding |
|---|---|
| `quantize_kat.py` | One commit in its entire history; unmodified since creation |
| `llm-compressor` / `transformers` | 0.13.0 / 5.14.1 — match the pins in `docs/environment.md` |
| `compressed-tensors` | 0.18.0, released before this project's first commit; no newer release exists |
| `reap` fork | HEAD is the router-renormalization fix, dated two minutes before this repo's first commit; zero commits behind origin |
| Prune parameters | ratio 0.50, seed 42, 64 batches/category, 2048 max length, calibration cache hit confirmed in the run log |

The discrepancy was then located directly, by parsing the `safetensors` header and
accounting for every byte by tensor group — the table above. This is the general
rule the project already follows for evaluation, applied to artifacts: a
checkpoint's declared contents and its actual tensor inventory are separate
things, and only the second is authoritative.

## Notes for reuse

Anyone building on `KAT-Coder-V2.5-Dev`, or on another checkpoint whose config
declares modules it does not ship, should expect the same behaviour. The tower is
invisible at inference — vLLM's `language_model_only=True` declines to load it, so
resident weights are ~12.58 GiB whether or not the tensors are present — which
means it costs download size and storage rather than VRAM, and will not surface as
a runtime error. It has to be checked for deliberately.
