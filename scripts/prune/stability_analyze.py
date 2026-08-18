"""Calibration size vs expert-ranking stability.

For each calibration size, two seeds draw different samples. The overlap between
the two resulting prune sets is the stability estimate: if 25% of experts are
removed and two independent calibrations pick the same ones, the ranking has
converged. If they disagree, any criteria comparison at that size is measuring
noise.

Reported per layer type as well, because published work notes that layers differ
markedly in their sensitivity to expert pruning.
"""

from __future__ import annotations

import glob
import statistics
import sys

import torch

ROOT = "/home/ttimm/reap-stability"
SEQLEN = 2048
SIZES = [4, 16, 64]
SEEDS = [42, 0]
PRUNE_FRAC = 0.25


def load(size: int, seed: int):
    hits = glob.glob(
        f"{ROOT}/n{size}_s{seed}/**/observations_*.pt", recursive=True
    )
    if not hits:
        return None
    state = torch.load(hits[0], map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    return state


def prune_set(layer_data, frac: float) -> set[int]:
    saliency = layer_data["reap"].float()
    k = max(1, int(frac * saliency.numel()))
    return set(torch.topk(saliency, k, largest=False).indices.tolist())


print(f"{'sequences':>10}{'tokens':>10}{'mean overlap':>14}{'min':>8}{'full_attn':>11}{'linear':>9}")
rows = []
for size in SIZES:
    a = load(size, SEEDS[0])
    b = load(size, SEEDS[1])
    if a is None or b is None:
        print(f"{size:>10}{size * SEQLEN:>10}   (incomplete)")
        continue

    overlaps, full_ov, lin_ov = [], [], []
    for layer in sorted(a, key=lambda x: int(x)):
        if layer not in b:
            continue
        sa, sb = prune_set(a[layer], PRUNE_FRAC), prune_set(b[layer], PRUNE_FRAC)
        ov = len(sa & sb) / max(len(sa), 1)
        overlaps.append(ov)
        # layer_types repeat 3x linear then 1x full
        (full_ov if int(layer) % 4 == 3 else lin_ov).append(ov)

    if not overlaps:
        continue
    row = (
        size,
        size * SEQLEN,
        statistics.mean(overlaps),
        min(overlaps),
        statistics.mean(full_ov) if full_ov else float("nan"),
        statistics.mean(lin_ov) if lin_ov else float("nan"),
    )
    rows.append(row)
    print(
        f"{row[0]:>10}{row[1]:>10}{row[2]:>13.1%}{row[3]:>8.1%}"
        f"{row[4]:>11.1%}{row[5]:>9.1%}"
    )

if len(rows) >= 2:
    print("\n=== convergence ===")
    for i in range(1, len(rows)):
        prev, cur = rows[i - 1], rows[i]
        gain = cur[2] - prev[2]
        print(
            f"{prev[0]:>3} -> {cur[0]:<3} sequences "
            f"({prev[1]:,} -> {cur[1]:,} tokens): overlap {prev[2]:.1%} -> {cur[2]:.1%} "
            f"({gain:+.1%})"
        )

    last_gain = rows[-1][2] - rows[-2][2]
    print("\n=== verdict ===")
    if rows[-1][2] >= 0.95 and last_gain < 0.02:
        print(f"Converged by {rows[-1][0]} sequences ({rows[-1][1]:,} tokens).")
        print("Calibration at or above this size gives a stable prune set.")
    elif last_gain < 0.02:
        print(f"Plateauing at only {rows[-1][2]:.1%} overlap. More calibration is")
        print("not fixing it; the ranking itself may be weakly determined.")
    else:
        print(f"Still climbing ({last_gain:+.1%} at the last step). Not yet converged;")
        print("extend the sweep to 128+ sequences before sizing the study.")
else:
    sys.exit("need at least two completed sizes")
