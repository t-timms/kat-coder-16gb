"""Build the W4A4 twin of the 13.28 GiB NVFP4A16 checkpoint.

THE QUESTION THIS ANSWERS
    The weight-only build runs at 8.9 tok/s because vLLM routes NVFP4A16 to the
    MARLIN kernel, which dequantises to bf16 - the native FP4 path (VLLM_CUTLASS)
    is unavailable to weight-only schemes. W4A4 unlocks it. But QSpec measured W4A4
    losing 38.73% on HumanEval where W4A16 barely moved (INT4-era data; NVFP4's
    per-16 block scaling should be far kinder).

    Same source checkpoint, same ignore list, ONLY the scheme differs, so the
    comparison is clean: speed vs coding accuracy, at identical file size.

DIFFERENCE FROM THE A16 RUN
    NVFP4A16 was inferred as a DataFreePipeline and took 82 s. W4A4 quantises
    ACTIVATIONS, so it needs real calibration data and will take substantially
    longer. Calibration uses evol-codealpaca, deliberately NOT the Magicoder set
    reserved for evaluation.
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
MAX_SEQ = 2048

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
