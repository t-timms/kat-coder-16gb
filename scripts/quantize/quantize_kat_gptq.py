"""Requantize the 50%-pruned KAT-Coder to NVFP4A16 using GPTQ instead of RTN.

WHY THIS RUN EXISTS
    The shipped checkpoint (quantize_kat.py) uses QuantizationModifier, which is
    plain round-to-nearest (RTN): each weight is rounded to its nearest
    representable NVFP4 value independently. GPTQModifier instead solves a
    layer-wise least-squares problem (approximate second-order / Hessian-based
    correction), accounting for how weights in the same layer interact, then
    corrects the rest of the layer for the error introduced by each rounding
    decision. Confirmed via direct verification, not assumption, before
    running this:
      - GPTQ+NVFP4 has been supported in llm-compressor since v0.10.0
        (2026-03-18 Red Hat release notes: "LLM Compressor v0.10: Faster
        compression with distributed GPTQ").
      - Literature comparison (multiple 2026 sources, checked 2026-08-22):
        GPTQ "consistently outperforms RTN" for NVFP4 recovery specifically,
        because RTN "applies per-element rounding independently, without
        leveraging the structural redundancy GPTQ uses for error compensation."
      - A newer method (MR-GPTQ, arXiv 2509.23202) claims better recovery
        still (98-99% of FP16), but is NOT usable here: its llm-compressor
        integration is an open, unimplemented RFC
        (github.com/vllm-project/llm-compressor/issues/2006) as of
        2026-08-22. Using it would mean bypassing this whole pipeline for a
        separate, unintegrated toolchain (FP-Quant/QuTLASS) - out of scope
        for this run. Tracked as a future watch item, not attempted here.

    Same file size as the shipped RTN build (weight-only, same NVFP4A16
    scheme, same targets/ignore list) - if this doesn't improve accuracy, it
    costs nothing to have tried, and the shipped checkpoint stays the
    default. This is not a replacement until the accuracy suite says so.

WHAT IS DELIBERATELY UNCHANGED FROM quantize_kat.py
    MAX_SEQ stays at 2048 (not the 4096 already applied to the untested
    W4A4 script) so this run isolates ONE variable - the quantization
    algorithm - against the shipped baseline. If GPTQ measures better,
    raising MAX_SEQ to match the 49K serving context is a legitimate
    follow-up, but compounding it here would make it impossible to
    attribute any accuracy change to GPTQ specifically.

IGNORE LIST
    Identical to quantize_kat.py. Verified directly against this exact
    checkpoint's tensor names before writing this script (not assumed): the
    router modules (`mlp.gate`, `shared_expert_gate`) are correctly matched
    by these patterns once compared at module-name granularity (llm-compressor
    matches ignore patterns against nn.Module names from named_modules(),
    which lack the trailing ".weight" a raw safetensors tensor name has - an
    easy category mismatch to get wrong when writing your own checker, and
    exactly what an earlier verification pass here did before being
    corrected). Ordinary per-expert SwiGLU gate_proj layers are correctly
    NOT ignored (they are meant to be quantized like any other Linear).

GPTQ-SPECIFIC HYPERPARAMETERS
    Written explicitly below rather than left as implicit defaults, so this
    recipe stays reproducible even if llm-compressor's own defaults change in
    a future version:
      - actorder="static": confirmed via 2026 sources as the setting that
        "achieves best accuracy recovery with no runtime cost" - not a
        speed/accuracy tradeoff, just strictly better.
      - block_size=128, dampening_frac=0.01: llm-compressor 0.13.0's own
        defaults, pinned explicitly here.
      - offload_hessians=False: llm-compressor 0.13.0's default. If this run
        OOMs during calibration, this is the first lever to flip (trades
        speed for memory) - not enabled preemptively since the memory
        profile of GPTQ on this exact checkpoint has not been measured yet.

CALIBRATION
    Same evol-codealpaca set, same size, same seed as quantize_kat.py -
    deliberately unchanged, for the same isolation reason as MAX_SEQ above.

KNOWN UNKNOWN, STATED PLAINLY
    GPTQ's Hessian-based correction is more expensive than RTN's independent
    rounding. How much slower on this exact 35B/3B-active checkpoint has not
    been measured - this script prints wall-clock time for both model load
    and the oneshot() call itself, same as quantize_kat.py, specifically so
    that number gets measured on the first real run instead of guessed at
    beforehand.
"""

from __future__ import annotations

import json
import pathlib
import time

import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

SRC = (
    pathlib.Path.home()
    / "reap-stability/n64_s42/model_--home--ttimm--models--KAT-Coder-V2.5-Dev-8ccb0b379945"
    / "dataset_theblackcat102--evol-codealpaca-v1-9d908ea05bb5/pruned_models"
    / "reap-renorm_true-seed_42-0.50"
)
DST = pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16-gptq"

NUM_CALIB = 256
MAX_SEQ = 2048

print(f"source : {SRC}", flush=True)
print(f"dest   : {DST}", flush=True)
if not SRC.is_dir():
    raise SystemExit(f"source checkpoint missing: {SRC}")

cfg = AutoConfig.from_pretrained(SRC, trust_remote_code=True)
archs = getattr(cfg, "architectures", None) or []
print(f"config : {cfg.__class__.__name__}  architectures={archs}", flush=True)

# Pick the auto class the checkpoint actually declares rather than assuming.
if any("ConditionalGeneration" in a or "ImageText" in a for a in archs):
    from transformers import AutoModelForImageTextToText as AutoCls

    print("using AutoModelForImageTextToText", flush=True)
else:
    from transformers import AutoModelForCausalLM as AutoCls

    print("using AutoModelForCausalLM", flush=True)

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
model = AutoCls.from_pretrained(
    SRC,
    dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
)
print(f"MODEL_LOADED in {time.time() - t0:.1f}s", flush=True)

ds = load_dataset("theblackcat102/evol-codealpaca-v1", split=f"train[:{NUM_CALIB * 2}]")
ds = ds.shuffle(seed=42).select(range(NUM_CALIB))


def preprocess(example):
    text = f"{example['instruction']}\n\n{example['output']}"
    return tokenizer(text, truncation=True, max_length=MAX_SEQ)


ds = ds.map(preprocess, remove_columns=ds.column_names)
print(f"calibration: {len(ds)} samples, max_seq {MAX_SEQ}", flush=True)

recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4A16",
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
    actorder="static",
    block_size=128,
    dampening_frac=0.01,
    offload_hessians=False,
)

t1 = time.time()
oneshot(
    model=model,
    # Pass the tokenizer explicitly. Without it llm-compressor calls AutoProcessor,
    # which tries to build Qwen3VLVideoProcessor for the phantom multimodal config
    # and dies on a missing torchvision. Installing torchvision would "fix" it by
    # satisfying a video pipeline for a model with zero vision weights; passing the
    # tokenizer says what is actually true, that this is a text-only quantization.
    processor=tokenizer,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQ,
    num_calibration_samples=NUM_CALIB,
    moe_calibrate_all_experts=True,
    output_dir=str(DST),
)
print(f"ONESHOT_DONE in {time.time() - t1:.1f}s", flush=True)

tokenizer.save_pretrained(DST)

# Carry over the processor files reap's save path drops; without them vLLM builds
# an image processor and dies on load.
base = pathlib.Path.home() / "models" / "KAT-Coder-V2.5-Dev"
for fname in (
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "chat_template.jinja",
):
    src_f = base / fname
    dst_f = DST / fname
    if src_f.is_file() and not dst_f.is_file():
        dst_f.write_bytes(src_f.read_bytes())
        print(f"  copied {fname}", flush=True)

# --- verify by artifact, never by exit code ---------------------------------
print("\n=== verification ===", flush=True)
shards = sorted(DST.glob("*.safetensors"))
total = sum(p.stat().st_size for p in shards)
print(f"  shards: {len(shards)}  total: {total / 2**30:.2f} GiB")

cfg_out = json.loads((DST / "config.json").read_text())
q = cfg_out.get("quantization_config", {})
print(f"  format: {q.get('format')}")
for gname, g in (q.get("config_groups") or {}).items():
    acts = g.get("input_activations")
    print(
        f"  {gname}: weights={g.get('weights', {}).get('num_bits')} "
        f"acts={acts.get('num_bits') if acts else 'None (weight-only)'}"
    )
tc = cfg_out.get("text_config", cfg_out)
print(f"  num_experts: {tc.get('num_experts')} (expect 128)")

if total / 2**30 > 15.0:
    print("  !! larger than 15 GiB - will not leave room for KV cache on a 16.3 GB card")
elif len(shards) == 0:
    print("  !! NO SHARDS WRITTEN")
else:
    print("  OK: fits a 16 GB card")
print("QUANTIZE_COMPLETE", flush=True)
