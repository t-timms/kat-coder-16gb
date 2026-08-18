"""Fix the router-renormalization gate in reap-cuda.

THE BUG
    pipeline.py decides whether to renormalize top-k router weights during saliency
    computation by asking the CONFIG:

        renormalize_router_weights = (
            getattr(model.config, "norm_topk_prob", False) and obs_args....
        )

    For Qwen3.5 MoE that attribute is ABSENT from config.json entirely, so the gate
    returns False. But the model renormalizes UNCONDITIONALLY at inference, hardcoded
    in Qwen3_5MoeTopKRouter.forward:

        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)

    So saliency was computed under routing semantics the model does not use, despite
    output directories named "reap-renorm_true-...". The metadata records the
    REQUESTED value, not the effective one, which is why the name lies.

    REAP's own ablation: 1.9% mean loss with renormalization vs 2.6% without. The
    router-calibration literature (arXiv 2603.02217) argues the step is necessary.

THE FIX
    Ask the ADAPTER what the architecture does, not the config. The adapter already
    encodes per-architecture knowledge, and the config demonstrably cannot answer.
    Default behaviour for every other adapter is unchanged (same getattr on config),
    so this is a safe, upstreamable narrowing.

This script verifies each anchor appears exactly once before editing, and refuses to
guess. Run it twice and the second run reports "already patched" rather than
corrupting the file.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path.home() / "reap-cuda" / "src" / "reap"
ADAPTERS = ROOT / "model_adapters.py"
PIPELINE = ROOT / "pipeline.py"

FAILURES: list[str] = []


def patch(path: pathlib.Path, anchor: str, replacement: str, marker: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"  already patched: {path.name}")
        return
    count = text.count(anchor)
    if count != 1:
        FAILURES.append(f"{path.name}: anchor found {count} times, expected exactly 1")
        return
    path.write_text(text.replace(anchor, replacement))
    print(f"  patched {path.name}")


# --- 1. adapters: declare what the architecture actually does ------------------
ADAPTER_ANCHOR = '''    adapter_name = "qwen3_5_moe"

    def hook_regex(self) -> str:
        return "Qwen3_5MoeSparseMoeBlock"'''

ADAPTER_NEW = '''    adapter_name = "qwen3_5_moe"

    def hook_regex(self) -> str:
        return "Qwen3_5MoeSparseMoeBlock"

    def renormalizes_router_weights(self, config) -> bool:
        """Qwen3.5/3.6 MoE always normalises top-k router weights.

        ``Qwen3_5MoeTopKRouter.forward`` does
        ``router_top_value /= router_top_value.sum(dim=-1, keepdim=True)``
        unconditionally, and ``norm_topk_prob`` is absent from these configs, so
        the usual ``getattr(config, "norm_topk_prob", False)`` probe answers False
        for a model that renormalises on every forward pass. Saliency must be
        computed under the routing semantics the model actually uses.
        """
        return True'''

# --- 2. pipeline: consult the adapter, fall back to the old probe ---------------
PIPELINE_ANCHOR = '''    renormalize_router_weights = (
        getattr(model.config, "norm_topk_prob", False)
        and obs_args.renormalize_router_weights
    )'''

PIPELINE_NEW = '''    # Ask the adapter what this architecture does; the config is not always able
    # to answer. Qwen3.5 MoE renormalises unconditionally in the router forward
    # while omitting norm_topk_prob from config.json, so the bare getattr probe
    # silently disables renormalisation for a model that always renormalises.
    _declared = getattr(adapter, "renormalizes_router_weights", None)
    if _declared is not None:
        _model_renormalizes = bool(_declared(model.config))
    else:
        _model_renormalizes = bool(getattr(model.config, "norm_topk_prob", False))

    renormalize_router_weights = (
        _model_renormalizes and obs_args.renormalize_router_weights
    )'''

print("=== patching ===")
patch(ADAPTERS, ADAPTER_ANCHOR, ADAPTER_NEW, "def renormalizes_router_weights")
patch(PIPELINE, PIPELINE_ANCHOR, PIPELINE_NEW, "_model_renormalizes")

if FAILURES:
    print("\n=== FAILED, nothing partially applied is safe ===")
    for f in FAILURES:
        print(f"  !! {f}")
    sys.exit(1)

# --- verify the result compiles and the adapter reports True -------------------
print("\n=== verification ===")
import py_compile  # noqa: E402

for p in (ADAPTERS, PIPELINE):
    py_compile.compile(str(p), doraise=True)
    print(f"  compiles: {p.name}")

sys.path.insert(0, str(ROOT.parent))
from reap.model_adapters import Qwen3_5MoeModelAdapter  # noqa: E402

adapter = Qwen3_5MoeModelAdapter()
got = adapter.renormalizes_router_weights(object())
print(f"  Qwen3_5MoeModelAdapter.renormalizes_router_weights(...) -> {got}")
if got is not True:
    print("  !! expected True")
    sys.exit(1)

print("\nOK. NOTE: this INVALIDATES every cached observations_*.pt, which was")
print("computed without renormalisation. Re-runs MUST use a fresh --artifacts-dir")
print("or reap will silently return the stale cached hit.")
