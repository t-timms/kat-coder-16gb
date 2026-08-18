"""Median and range for the A/B latency benchmark, plus the kernel each arm used.

Reports MEDIAN AND RANGE over separate process invocations, never a single number.
Batch-1 decode on this machine once measured 9.6 / 32.2 / 23.7 tok/s across three
runs of an identical command, each internally tight to 0.2-1.8% - a 3.4x range. So
a lone figure, however precise it looks, is not a result.

The warmup rep (rep0) is excluded: a cold compile cache measures the compiler.
"""

from __future__ import annotations

import json
import re
import statistics as stats
from pathlib import Path

OUT = Path.home() / "bench-ab"
OUTPUT_LEN = 256

ARMS = {
    "a16": "NVFP4A16 (weight-only, Marlin dequant)",
    "w4a4": "NVFP4 W4A4 (native FP4)",
}


def latencies(tag: str) -> list[float]:
    vals = []
    for p in sorted(OUT.glob(f"{tag}_rep*.json")):
        if p.stem.endswith("rep0"):  # warmup, discarded
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        v = d.get("avg_latency")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return vals


def kernel_for(tag: str) -> str:
    for p in sorted(OUT.glob(f"{tag}_rep*.log")):
        m = re.search(r"Using '([A-Z_]+)' NvFp4 MoE backend", p.read_text(errors="ignore"))
        if m:
            return m.group(1)
    return "unknown"


results = {}
print("=== A/B latency: median and range over separate invocations ===\n")

for tag, label in ARMS.items():
    vals = latencies(tag)
    kern = kernel_for(tag)
    print(f"{label}")
    print(f"  kernel selected : {kern}")
    if not vals:
        print("  !! no results\n")
        continue
    med = stats.median(vals)
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / med * 100 if med else 0
    tps = OUTPUT_LEN / med if med else 0
    print(f"  n invocations   : {len(vals)}")
    print(f"  latency median  : {med:.3f} s   range [{lo:.3f}, {hi:.3f}]  spread {spread:.1f}%")
    print(f"  decode tok/s    : {tps:.1f}  (median, batch=1, {OUTPUT_LEN} out tokens)")
    if len(vals) < 5:
        print(f"  ⚠ only {len(vals)} invocations; >=5 required before quoting")
    print()
    results[tag] = (med, tps, kern)

if len(results) == 2:
    (ma, ta, ka), (mb, tb, kb) = results["a16"], results["w4a4"]
    print("=== verdict ===")
    print(f"  a16  {ta:6.1f} tok/s via {ka}")
    print(f"  w4a4 {tb:6.1f} tok/s via {kb}")
    if ta > 0:
        print(f"  W4A4 speedup: {tb / ta:.2f}x")
    if ka == kb:
        print(f"  ⚠ BOTH arms used {ka} - the kernel hypothesis is NOT being tested")
    print()
    print("  Speed alone does not decide this. W4A4 must also be measured on")
    print("  HumanEval: QSpec reported W4A4 losing 38.73% there while W4A16 barely")
    print("  moved. Same file size either way, so the choice is speed vs accuracy.")
