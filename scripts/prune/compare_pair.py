"""Compare layerwise observations against the cpu_full baseline.

Same model, same dataset, same seed, same 16x256 calibration. The two paths
compute the same statistics by different routes, so agreement is the real test
that the layerwise fixes are correct rather than merely non-crashing.

A fast path that produces different saliency is worse than no fast path.
"""

from __future__ import annotations

import glob
import statistics
import sys

import torch

BASE = __import__("sys").argv[1]
NEW = __import__("sys").argv[2]


def load(root: str):
    hits = glob.glob(f"{root}/**/observations_*.pt", recursive=True)
    if not hits:
        sys.exit(f"no observation artifact under {root}")
    state = torch.load(hits[0], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    return hits[0], state


base_path, base = load(BASE)
new_path, new = load(NEW)
print(f"baseline (cpu_full) : {base_path}")
print(f"candidate (layerwise): {new_path}\n")

base_layers = sorted(base)
new_layers = sorted(new)
print(f"layers: baseline={len(base_layers)} candidate={len(new_layers)}")
if base_layers != new_layers:
    print("!! layer sets differ")

worst_rel = 0.0
worst_layer = None
rank_agree = []

for layer in base_layers:
    if layer not in new:
        print(f"!! layer {layer} missing from candidate")
        continue
    b = base[layer]
    n = new[layer]

    # Routing counts must match closely: same data, same seed.
    bf = b["expert_frequency"].float()
    nf = n["expert_frequency"].float()
    denom = bf.abs().sum().clamp_min(1e-9)
    rel = (bf - nf).abs().sum() / denom
    if rel > worst_rel:
        worst_rel, worst_layer = float(rel), layer

    # What actually drives pruning is the ORDER of experts by saliency.
    if "reap" in b and "reap" in n:
        bs = b["reap"].float()
        ns = n["reap"].float()
        k = max(1, int(0.25 * bs.numel()))
        b_pruned = set(torch.topk(bs, k, largest=False).indices.tolist())
        n_pruned = set(torch.topk(ns, k, largest=False).indices.tolist())
        rank_agree.append(len(b_pruned & n_pruned) / k)

print(f"\nworst relative routing difference: {worst_rel:.4%} (layer {worst_layer})")

if rank_agree:
    print("\n=== agreement on WHICH experts get pruned at 25% ===")
    print(f"layers compared : {len(rank_agree)}")
    print(f"mean overlap    : {statistics.mean(rank_agree):.2%}")
    print(f"min overlap     : {min(rank_agree):.2%}")
    print(f"perfect layers  : {sum(1 for a in rank_agree if a == 1.0)}/{len(rank_agree)}")

print("\n=== verdict ===")
if worst_rel < 0.01 and rank_agree and min(rank_agree) > 0.95:
    print("Layerwise agrees with cpu_full. The fast path is also the correct path.")
elif rank_agree and statistics.mean(rank_agree) > 0.9:
    print("Close but not identical. Investigate before trusting layerwise output.")
else:
    print("DISAGREES. Layerwise output must not be used for published pruning.")
