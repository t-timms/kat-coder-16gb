"""Build the W4A4 twin of the 13.28 GiB NVFP4A16 checkpoint.

THE QUESTION THIS ANSWERS
    vLLM routes NVFP4A16 to the MARLIN kernel, which decodes 4-bit weights and
    computes in bf16; the native FP4 GEMM paths are reachable only by schemes that
    also quantise activations. W4A4 reaches them.

    On the motivation, honestly: this script was written when the weight-only build
    measured 8.9 tok/s, BEFORE CUDA graphs were found to work on this card. They
    do, and the same weight-only build now serves at 149.5 tok/s, so W4A4 is not a
    rescue from an unusable baseline. The open question is narrower and still worth
    answering: does a native FP4 path beat Marlin plus CUDA graphs at all, and what
    does it cost in accuracy? QSpec measured W4A4 losing 38.73% on HumanEval where
    W4A16 barely moved (INT4-era data; NVFP4's per-16 block scaling should be far
    kinder).

    Same source checkpoint, same ignore list, ONLY the scheme differs, so the
    comparison is clean: speed vs coding accuracy, at identical file size.

DIFFERENCE FROM THE A16 RUN
    NVFP4A16 was inferred as a DataFreePipeline and took 82 s. W4A4 quantises
    ACTIVATIONS, so it needs real calibration data and will take substantially
    longer. Calibration uses evol-codealpaca, deliberately NOT the Magicoder set
    reserved for evaluation.

PRE-LAUNCH AUDIT (2026-08-21, no GPU/CPU compute spent — filesystem + docs only)
    Checked this recipe and this box's environment against upstream llm-compressor's
    own Qwen3.5 MoE NVFP4 example and against the vLLM/FlashInfer source actually
    installed here, before spending a CPU-hours-long calibration run on it:
      - Ignore list matches the official Qwen3.5-MoE NVFP4 example exactly (lm_head,
        visual*, mlp.gate$, embed_tokens$, shared_expert_gate$, linear_attn*), plus
        two extras this checkpoint's architecture needs (conv1d*, mtp*) that the
        stock example doesn't have to handle.
      - scheme="NVFP4" defaults (compressed-tensors 0.18.0, installed in
        ~/quant-env) already ARE current best practice with no recipe override
        needed: weights use memoryless_minmax, input_activations use dynamic-local
        + static_minmax — matches what upstream now ships by default. GPTQModifier
        is NOT part of the reference recipe for NVFP4 (unlike coarser INT4
        schemes) — not added here.
      - MAX_SEQ raised 2048 -> 4096 to match the official Qwen3.5-MoE NVFP4 example
        (256 samples / 4096 tokens) and because this project's SWE-bench serving
        config now runs a 49,152-token window — the old 2048 cap calibrated on
        much shorter contexts than deployment actually uses. Box has 78 GiB RAM
        free (checked 2026-08-21), so this is affordable; it does roughly double
        CPU calibration wall-clock, since `device_map="cpu"` runs calibration
        forward passes on CPU, not GPU.
      - The critical SM12x kernel bug this scheme could hit (vLLM PR #35947/#37725:
        CMake was stripping the "a" arch suffix, producing plain sm_120 instead of
        sm_120a and disabling the native e2m1 FP4 conversion instruction, causing
        NaN in NVFP4 activation quantization) is CONFIRMED ALREADY FIXED in this
        box's vLLM build: `~/vllm-src` is tag v0.26.0 (2026-07-26), and
        `cmake/utils.cmake`'s `string_to_ver` macro already preserves the a/f
        suffix, with `CUDA_SUPPORTED_ARCHS` including 12.0 and 12.1. Nothing to
        patch.
      - All four "Critical priority" lna-lab/blackwell-geforce-nvfp4-gemm patches
        that matter for MoE FP4 on SM120 (#1 grouped-GEMM tile sizing, #3/#4
        device-family gates, #8/#9 FlashInfer JIT SM120 gencode + FP4-quant JIT)
        are CONFIRMED already present in the installed flashinfer 0.6.14 and this
        vllm-src checkout — verified directly by grepping the installed source,
        not inferred. No community patch needs applying before this build's first
        serve.
      - `~/.cache/flashinfer` already exists (180 MiB, populated by the A16 runs)
        and FlashInfer caches compiled kernels there across server restarts. The
        833 s first-load JIT cost measured 2026-08-20 is a one-time cost FOR THIS
        MACHINE, not per-restart — it just hasn't been paid yet for W4A4's own
        kernel set (activation-quant + FP4 MoE grouped GEMM), since A16 never
        exercises those kernels. Consider `VLLM_USE_AOT_COMPILE=1` (exists in this
        vLLM build, default off) to test whether it removes the JIT cost outright
        rather than just caching it after the first hit.
      - NOT verified: an actual W4A4 MoE forward pass has never been run on this
        card. The FlashInfer CUTLASS MoE + runtime activation-quantization
        combination is new to this box even though its individual pieces check
        out on paper. Per this project's own standing practice, smoke-test on a
        handful of tokens before spending the full HumanEval+/MBPP+ suite on it.
"""

from __future__ import annotations

import json
import pathlib
import time

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SRC = (
    pathlib.Path.home()
    / "reap-stability/n64_s42/model_--home--ttimm--models--KAT-Coder-V2.5-Dev-8ccb0b379945"
    / "dataset_theblackcat102--evol-codealpaca-v1-9d908ea05bb5/pruned_models"
    / "reap-renorm_true-seed_42-0.50"
)
DST = pathlib.Path.home() / "models" / "kat-50pct-nvfp4-w4a4"

NUM_CALIB = 256
MAX_SEQ = 4096  # raised from 2048 2026-08-21; see PRE-LAUNCH AUDIT above

print(f"source : {SRC}", flush=True)
print(f"dest   : {DST}", flush=True)
if not SRC.is_dir():
    raise SystemExit(f"source checkpoint missing: {SRC}")

cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
archs = getattr(cfg, "architectures", None) or []
print(f"config : {cfg.__class__.__name__}  architectures={archs}", flush=True)

if any("ConditionalGeneration" in a or "ImageText" in a for a in archs):
    from transformers import AutoModelForImageTextToText as AutoCls
else:
    from transformers import AutoModelForCausalLM as AutoCls
print(f"auto class: {AutoCls.__name__}", flush=True)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoCls.from_pretrained(
    SRC, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
)
print(f"MODEL_LOADED in {time.time() - t0:.1f}s", flush=True)

ds = load_dataset("theblackcat102/evol-codealpaca-v1", split=f"train[:{NUM_CALIB * 4}]")
ds = ds.shuffle(seed=42).select(range(NUM_CALIB))


def preprocess(example):
    return tokenizer(
        f"{example['instruction']}\n\n{example['output']}",
        truncation=True,
        max_length=MAX_SEQ,
    )


ds = ds.map(preprocess, remove_columns=ds.column_names)
print(f"calibration: {len(ds)} samples, max_seq {MAX_SEQ}", flush=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",  # W4A4: weights AND activations to 4 bits
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
        "re:.*conv1d.*",
        "re:.*mtp.*",
    ],
)

t1 = time.time()
oneshot(
    model=model,
    processor=tokenizer,  # else AutoProcessor builds Qwen3VLVideoProcessor and dies
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQ,
    num_calibration_samples=NUM_CALIB,
    moe_calibrate_all_experts=True,
    output_dir=str(DST),
)
print(f"ONESHOT_DONE in {time.time() - t1:.1f}s", flush=True)

tokenizer.save_pretrained(DST)

base = pathlib.Path.home() / "models" / "KAT-Coder-V2.5-Dev"
for fname in (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
):
    src_f, dst_f = base / fname, DST / fname
    if src_f.is_file() and not dst_f.is_file():
        dst_f.write_bytes(src_f.read_bytes())
        print(f"  copied {fname}", flush=True)

print("\n=== verification ===", flush=True)
shards = sorted(DST.glob("*.safetensors"))
total = sum(p.stat().st_size for p in shards)
print(f"  shards: {len(shards)}  total: {total / 2**30:.2f} GiB")

cfg_out = json.loads((DST / "config.json").read_text())
q = cfg_out.get("quantization_config", {})
print(f"  format: {q.get('format')}")
acts_ok = False
for gname, g in (q.get("config_groups") or {}).items():
    acts = g.get("input_activations")
    nbits = acts.get("num_bits") if acts else None
    print(f"  {gname}: weights={g.get('weights', {}).get('num_bits')} acts={nbits}")
    if nbits == 4:
        acts_ok = True
tc = cfg_out.get("text_config", cfg_out)
print(f"  num_experts: {tc.get('num_experts')} (expect 128)")

# The whole point of this build is 4-bit ACTIVATIONS. Grepping num_bits:4 cannot
# tell W4A4 from W4A16 - only input_activations decides it.
if not acts_ok:
    print("  !! input_activations is NOT 4-bit - this is not a W4A4 build")
elif len(shards) == 0:
    print("  !! NO SHARDS WRITTEN")
elif total / 2**30 > 15.0:
    print("  !! larger than 15 GiB - no room for KV cache")
else:
    print("  OK: W4A4 confirmed and fits a 16 GB card")
print("QUANTIZE_COMPLETE", flush=True)
