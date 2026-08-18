"""Quantize the 50%-pruned KAT-Coder to NVFP4 so it fits 16 GB natively.

WHY THIS RUN EXISTS
    The 36 GB bf16 pruned checkpoint cannot fit 16.3 GB of VRAM, so every test so
    far streamed weights over PCIe. That path just crashed in vllm's CPU-offload
    UVA code (uva.py:119, illegal memory access) before the NVFP4 kernels were ever
    reached. A ~11-12 GB NVFP4 checkpoint fits natively, removes the offloader from
    the picture entirely, and is the first real test of FP4 compute on SM120 for
    qwen3_5_moe. It is also the product.

SCHEME: NVFP4A16, not NVFP4
    NVFP4 in compressed-tensors means W4A4. NVFP4A16 is weight-only.
    Both give the SAME file size, because activations are never stored. W4A4 buys
    ~31% throughput and risks the capability this model exists for: QSpec measured
    W4A4 losing 38.73% on HumanEval where W4A16 barely moved. That is INT4-era data
    and NVFP4's per-16 block scaling is far better, so it is a warning rather than
    a verdict - but when the safer option costs nothing in size, take it first and
    measure W4A4 against it afterwards.

IGNORE LIST
    Follows llm-compressor's official Qwen3.5 NVFP4 MoE recipe, which matters
    because KAT is 3:1 hybrid linear attention (linear_attn must not be quantized)
    and carries a phantom vision tower with zero trained weights.

CALIBRATION
    evol-codealpaca, deliberately NOT the Magicoder set used for evaluation, so the
    eval corpus stays uncontaminated.
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
DST = pathlib.Path.home() / "models" / "kat-50pct-nvfp4a16"

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

recipe = QuantizationModifier(
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
