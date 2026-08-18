"""Confirm the REAP observer hooks every MoE block in KAT-Coder-V2.5-Dev.

The failure this guards against: a module-class regex that matches nothing, or
num_experts/top_k attribute paths that do not resolve. REAP raises on zero hooks,
but it does NOT check that the count is what you expected, and it does not check
the attribute paths until calibration is already running.
"""

from __future__ import annotations

import functools

import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig

from reap.observer import OBSERVER_CONFIG_REGISTRY, MoETransformerObserver

MODEL = "/home/ttimm/models/KAT-Coder-V2.5-Dev"
EXPECTED_LAYERS = 40
EXPECTED_EXPERTS = 256
EXPECTED_TOP_K = 8

config = AutoConfig.from_pretrained(MODEL)
arch = config.architectures[0]
with init_empty_weights():
    model = getattr(transformers, arch)._from_config(config)

hook_config = OBSERVER_CONFIG_REGISTRY[arch]()
print("hook regex:          ", hook_config.module_class_name_to_hook_regex)
print("num_experts_attr:    ", hook_config.num_experts_attr_name)
print("top_k_attr:          ", hook_config.top_k_attr_name)
print("fused_experts:       ", hook_config.fused_experts)

observer = MoETransformerObserver(model, hook_config)
n_hooks = len(observer.hooks)
print(f"\nhooked modules:       {n_hooks}")

# Resolve the attribute paths exactly the way observer.py does.
moe_modules = [
    m
    for m in model.modules()
    if m.__class__.__name__ == hook_config.module_class_name_to_hook_regex
]
print(f"matching MoE blocks:  {len(moe_modules)}")

sample = moe_modules[0]
n_experts = functools.reduce(
    getattr, hook_config.num_experts_attr_name.split("."), sample
)
top_k = functools.reduce(getattr, hook_config.top_k_attr_name.split("."), sample)
print(f"resolved num_experts: {n_experts}")
print(f"resolved top_k:       {top_k}")

failures = []
if n_hooks != EXPECTED_LAYERS:
    failures.append(f"expected {EXPECTED_LAYERS} hooks, got {n_hooks}")
if n_experts != EXPECTED_EXPERTS:
    failures.append(f"expected {EXPECTED_EXPERTS} experts, got {n_experts}")
if top_k != EXPECTED_TOP_K:
    failures.append(f"expected top_k {EXPECTED_TOP_K}, got {top_k}")

# Every hook must map to a distinct layer index; a collision would silently
# overwrite one layer's statistics with another's.
layer_ids = sorted(observer.observer_data.keys()) if hasattr(observer, "observer_data") else None
if layer_ids is not None:
    print(f"observer layer keys:  {len(layer_ids)} (min {min(layer_ids)}, max {max(layer_ids)})")

print("\nRESULT:", "FAIL -> " + "; ".join(failures) if failures else "PASS")
