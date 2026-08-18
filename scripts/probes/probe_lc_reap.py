"""Does llm-compressor's REAP support Qwen3.5 MoE out of the box?

This settles the fork-versus-migrate question with evidence instead of inference.

llm-compressor detects MoE layers by DUCK TYPING, not an architecture allowlist:
it looks for a module carrying both an experts attr (name in EXPERTS_ATTRS) and a
router attr (name in ROUTER_ATTRS), and requires the experts module to be
transformers' generic LinearExperts2D. If Qwen3_5MoeExperts qualifies, the whole
MODEL_ATTRS entry we hand-wrote in the ~/reap fork is unnecessary.

Builds the model on meta device, so this costs seconds rather than loading 69 GB.
"""

from __future__ import annotations

import os
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor.modifiers.pruning.reap.utils import EXPERTS_ATTRS, ROUTER_ATTRS

MODEL = os.path.expanduser("~/models/KAT-Coder-V2.5-Dev")

print(f"EXPERTS_ATTRS: {EXPERTS_ATTRS}")
print(f"ROUTER_ATTRS : {ROUTER_ATTRS}")

try:
    from transformers.masking_utils import LinearExperts2D  # noqa: F401

    print("LinearExperts2D imported from transformers.masking_utils")
except ImportError:
    from llmcompressor.modifiers.pruning.reap.utils import LinearExperts2D

    print("LinearExperts2D imported via llmcompressor")

cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
print(f"\nconfig class: {cfg.__class__.__name__}")

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

print(f"model class : {model.__class__.__name__}")

hits = []
for name, module in model.named_modules():
    e_attr = next((a for a in EXPERTS_ATTRS if hasattr(module, a)), None)
    r_attr = next((a for a in ROUTER_ATTRS if hasattr(module, a)), None)
    if e_attr and r_attr:
        experts = getattr(module, e_attr)
        hits.append((name, e_attr, r_attr, type(experts).__name__,
                     isinstance(experts, LinearExperts2D)))

print(f"\nmodules with BOTH experts+router: {len(hits)}")
if not hits:
    print("  !! none - llm-compressor REAP would raise ValueError on this model")
else:
    name, e_attr, r_attr, cls, is_l2d = hits[0]
    print(f"  first: {name}")
    print(f"    experts attr : {e_attr}  -> {cls}")
    print(f"    router  attr : {r_attr}")
    print(f"    isinstance(LinearExperts2D): {is_l2d}")
    n_ok = sum(1 for h in hits if h[4])
    print(f"\n  layers passing the LinearExperts2D check: {n_ok}/{len(hits)}")
    if n_ok == len(hits) and n_ok > 0:
        print("\n  VERDICT: SUPPORTED out of the box. The ~/reap fork is redundant.")
    elif n_ok == 0:
        print("\n  VERDICT: NOT supported - experts are not LinearExperts2D.")
        print("  The fork stays, or llm-compressor needs a patch.")
    else:
        print("\n  VERDICT: PARTIAL - some layers would be silently skipped.")
        print("  That is the dangerous case: it would prune fewer layers than asked.")
