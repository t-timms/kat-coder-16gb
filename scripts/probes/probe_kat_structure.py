"""Probe the in-memory module tree of KAT-Coder-V2.5-Dev.

Builds the model on the meta device with empty weights, so this costs seconds and
no VRAM instead of materialising 69.3 GB. Everything the REAP patch assumes is
checked here BEFORE any pruning run, because a wrong module/attribute name makes
the observer hook nothing and silently produce empty statistics rather than error.
"""

from __future__ import annotations

import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig

from reap.model_util import MODEL_ATTRS, get_config, get_layers, get_moe
from reap.observer import OBSERVER_CONFIG_REGISTRY

MODEL = "/home/ttimm/models/KAT-Coder-V2.5-Dev"


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


rule("versions")
print("transformers", transformers.__version__)

rule("config")
config = AutoConfig.from_pretrained(MODEL)
print("config class:", type(config).__name__)
print("model_type:", config.model_type)
print("architectures:", config.architectures)
print("has text_config:", hasattr(config, "text_config"))
print("top-level num_experts:", getattr(config, "num_experts", "ABSENT"))
print("text_config.num_experts:", getattr(config.text_config, "num_experts", "ABSENT"))
print(
    "text_config.num_experts_per_tok:",
    getattr(config.text_config, "num_experts_per_tok", "ABSENT"),
)

rule("modeling module class names")
mod = transformers.models.qwen3_5_moe.modeling_qwen3_5_moe
names = [
    n
    for n in dir(mod)
    if any(k in n for k in ("Moe", "Expert", "Router"))
    and isinstance(getattr(mod, n), type)
]
for n in sorted(names):
    print(" ", n)

rule("building empty model")
arch = config.architectures[0]
cls = getattr(transformers, arch)
with init_empty_weights():
    model = cls._from_config(config)
print("model class:", model.__class__.__name__)
print("registered in MODEL_ATTRS:", model.__class__.__name__ in MODEL_ATTRS)
print("registered in OBSERVER_CONFIG_REGISTRY:", model.__class__.__name__ in OBSERVER_CONFIG_REGISTRY)

attrs = MODEL_ATTRS[model.__class__.__name__]

rule("layers path resolution")
layers = get_layers(model)
print("layers_path:", attrs["layers_path"])
print("num layers:", len(layers))

rule("per-layer mlp class (checking for dense layers)")
mlp_classes = {}
for i in range(len(layers)):
    name = getattr(layers[i], attrs["moe_block"]).__class__.__name__
    mlp_classes.setdefault(name, []).append(i)
for name, idxs in mlp_classes.items():
    preview = idxs[:5]
    print(f"  {name}: {len(idxs)} layers, e.g. {preview}")

rule("MoE block internals (layer 0)")
moe = get_moe(model, 0)
print("moe class:", moe.__class__.__name__)
print("observer regex:", OBSERVER_CONFIG_REGISTRY[model.__class__.__name__].module_class_name_to_hook_regex)
print("moe has num_experts attr:", hasattr(moe, "num_experts"))
print("moe has top_k attr:", hasattr(moe, "top_k"))
print("moe children:", [n for n, _ in moe.named_children()])

rule("experts")
experts = getattr(moe, attrs["experts"])
print("experts class:", experts.__class__.__name__)
print("is ModuleList:", isinstance(experts, __import__("torch").nn.ModuleList))
params = [n for n, _ in experts.named_parameters(recurse=False)]
print("direct params:", params)
for key in ("gate_proj", "up_proj", "down_proj"):
    name = attrs[key]
    present = hasattr(experts, name)
    shape = tuple(getattr(experts, name).shape) if present else None
    print(f"  attrs[{key}] -> {name}: present={present} shape={shape}")
print("experts.num_experts:", getattr(experts, "num_experts", "ABSENT"))

rule("router")
router = getattr(moe, attrs["router"])
print("router class:", router.__class__.__name__)
print("router children:", [n for n, _ in router.named_children()])
print("router direct params:", [n for n, _ in router.named_parameters(recurse=False)])
for a in ("weight", "bias", "out_features", "num_experts", "top_k"):
    has = hasattr(router, a)
    val = None
    if has and a in ("out_features", "num_experts", "top_k"):
        val = getattr(router, a)
    if has and a in ("weight", "bias"):
        w = getattr(router, a)
        val = tuple(w.shape) if w is not None else None
    print(f"  router.{a}: present={has} value={val}")

rule("get_config resolution")
cfg = get_config(model)
print("resolved config class:", type(cfg).__name__)
print("resolved num_experts:", getattr(cfg, attrs["num_experts"], "ABSENT"))

rule("DONE")
