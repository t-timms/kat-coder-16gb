"""Read every score from the accuracy suite, with binomial confidence intervals.

The driver's grep found no pass@1 for mbpp_plus, which means either the metric is
named differently or the run produced nothing. Reading the JSON settles which.

Confidence intervals matter here: 164 problems is a small sample. A pass@1 of 0.89
carries roughly +/-4.8 points at 95%, so quoting it to four decimals would imply a
precision the sample cannot support.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path.home() / "eval-suite"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval: better than normal approximation near 0 or 1."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


for task_dir in sorted(ROOT.iterdir()):
    if not task_dir.is_dir():
        continue
    results = sorted(task_dir.rglob("results_*.json"))
    samples = sorted(task_dir.rglob("samples_*.jsonl"))

    print(f"\n=== {task_dir.name} ===")
    if not results:
        print("  !! no results json")
        continue

    data = json.loads(results[0].read_text())
    n = len(samples[0].read_text().splitlines()) if samples else 0

    for task, metrics in (data.get("results") or {}).items():
        for key, val in metrics.items():
            if key == "alias" or not isinstance(val, (int, float)):
                continue
            line = f"  {key:24s} {val:.4f}"
            if 0.0 <= val <= 1.0 and n:
                lo, hi = wilson(round(val * n), n)
                line += f"   n={n}  95% CI [{lo * 100:.1f}, {hi * 100:.1f}]"
                line += f"   = {val * 100:.1f}%"
            print(line)
