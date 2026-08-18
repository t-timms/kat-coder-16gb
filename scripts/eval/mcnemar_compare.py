"""Paired McNemar test: does the calibration draw change model quality?

Two checkpoints differing only in which 64 calibration samples were drawn, and
disagreeing on ~9 experts per layer. Evaluated on the SAME items, so the
comparison is paired and McNemar is the correct test. An unpaired test on
aggregate accuracies throws away the pairing and wastes the run.

Discordance is the binding quantity, not accuracy. Published guidance: roughly
40-50 discordant pairs are needed for 80% power at alpha=0.05. This reports the
discordant count first so an underpowered result is visible as underpowered
rather than reported as a null.
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path("/home/ttimm/reap-eval")
METRIC_KEYS = ("acc_norm", "acc")


def load_samples(seed: int) -> dict[str, int]:
    """Return {doc_id: correct} for one seed's logged per-item outcomes."""
    hits = sorted(glob.glob(str(ROOT / f"seed{seed}" / "**" / "samples_*.jsonl"),
                            recursive=True))
    if not hits:
        sys.exit(f"no samples_*.jsonl under {ROOT}/seed{seed} (was --log_samples set?)")

    outcomes: dict[str, int] = {}
    with open(hits[-1]) as fh:
        for line in fh:
            rec = json.loads(line)
            doc_id = str(rec.get("doc_id", rec.get("doc_hash")))
            got = None
            for key in METRIC_KEYS:
                if key in rec:
                    got = rec[key]
                    break
            if got is None:
                continue
            outcomes[doc_id] = int(round(float(got)))
    if not outcomes:
        sys.exit(f"parsed no outcomes from {hits[-1]}")
    print(f"seed {seed}: {len(outcomes)} items from {Path(hits[-1]).name}")
    return outcomes


a = load_samples(42)
b = load_samples(0)

shared = sorted(set(a) & set(b))
if not shared:
    sys.exit("no shared items between the two runs; cannot pair")
print(f"\npaired on {len(shared)} shared items")

# Contingency
both_right = sum(1 for d in shared if a[d] and b[d])
both_wrong = sum(1 for d in shared if not a[d] and not b[d])
only_42 = sum(1 for d in shared if a[d] and not b[d])
only_0 = sum(1 for d in shared if not a[d] and b[d])
discordant = only_42 + only_0

acc_42 = sum(a[d] for d in shared) / len(shared)
acc_0 = sum(b[d] for d in shared) / len(shared)

print("\n=== accuracies ===")
print(f"  seed 42 : {acc_42:.4f}")
print(f"  seed 0  : {acc_0:.4f}")
print(f"  delta   : {acc_42 - acc_0:+.4f} ({(acc_42 - acc_0) * 100:+.2f} pp)")

print("\n=== paired contingency ===")
print(f"  both correct        : {both_right}")
print(f"  both wrong          : {both_wrong}")
print(f"  only seed 42 correct: {only_42}")
print(f"  only seed 0 correct : {only_0}")
print(f"  DISCORDANT          : {discordant}")

print("\n=== power check ===")
if discordant < 40:
    print(f"  {discordant} discordant pairs is UNDERPOWERED (need ~40-50 for 80%")
    print("  power at alpha=0.05). Extend --limit before drawing any conclusion.")
    print("  A null result here means 'not enough data', NOT 'no difference'.")
else:
    print(f"  {discordant} discordant pairs is adequate (need ~40-50).")

# Exact binomial McNemar, appropriate at these counts (no continuity fudge).
if discordant == 0:
    print("\n=== verdict ===")
    print("  Zero discordant pairs: the two checkpoints answered every item")
    print("  identically. Strong evidence the calibration draw is irrelevant here.")
    sys.exit(0)

n, k = discordant, min(only_42, only_0)
p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))

print("\n=== McNemar exact (two-sided) ===")
print(f"  n discordant = {n}, smaller cell = {k}")
print(f"  p = {p:.4g}")

print("\n=== verdict ===")
if p < 0.05:
    better = "seed 42" if only_42 > only_0 else "seed 0"
    print(f"  SIGNIFICANT (p={p:.4g}). {better} is genuinely better.")
    print("  Reading A: the calibration draw materially changes model quality.")
    print("  Pruning results are sensitive to which samples were drawn, and any")
    print("  criteria comparison must control for it.")
elif discordant >= 40:
    print(f"  NOT significant (p={p:.4g}) with adequate power.")
    print("  Reading B: expert choice is underdetermined but HARMLESS. Many")
    print("  different prune sets are about equally good, which is itself a")
    print("  useful practical result: stop optimising calibration size past this")
    print("  point.")
else:
    print(f"  NOT significant (p={p:.4g}) but UNDERPOWERED. Inconclusive.")
    print("  Do not report this as evidence of no difference.")
