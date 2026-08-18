"""Probe every held-out code corpus candidate in ONE pass.

The eval corpus must be independent of theblackcat102/evol-codealpaca-v1, because
a calibration document leaking into the eval set would give whichever seed drew it
an artificially lower perplexity on it, biasing the exact comparison being run.

Asks the Hub for configs, splits, row counts and gating without downloading data.
Discovering these serially through the loader costs a failed download per mistake.
"""

from __future__ import annotations

from datasets import (
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset_builder,
)

CANDIDATES = [
    # instruction-style code, same flavour as the calibration set, different source
    "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "m-a-p/CodeFeedback-Filtered-Instruction",
    # raw source files
    "bigcode/the-stack-smol",
    # small canonical code benchmarks, useful as secondary
    "google-research-datasets/mbpp",
    "openai/openai_humaneval",
]

for name in CANDIDATES:
    print(f"\n=== {name} ===")
    try:
        configs = get_dataset_config_names(name)
        print(f"  configs: {configs[:8]}{' ...' if len(configs) > 8 else ''}")
    except Exception as exc:  # noqa: BLE001
        print(f"  UNAVAILABLE -> {str(exc).splitlines()[0][:160]}")
        continue

    for config in configs[:3]:
        try:
            splits = get_dataset_split_names(name, config)
        except Exception as exc:  # noqa: BLE001
            print(f"    [{config}] splits ERROR -> {str(exc).splitlines()[0][:120]}")
            continue

        rows = {}
        try:
            info = load_dataset_builder(name, config).info
            if info.splits:
                rows = {k: v.num_examples for k, v in info.splits.items()}
        except Exception as exc:  # noqa: BLE001
            rows = {"?": str(exc).splitlines()[0][:60]}

        print(f"    [{config}] splits={splits} rows={rows}")

        try:
            feats = load_dataset_builder(name, config).info.features
            if feats:
                print(f"      fields: {list(feats.keys())}")
        except Exception:  # noqa: BLE001
            pass
