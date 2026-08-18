"""What the SWE-bench pilot actually did, before anything is scored.

The pilot's job is not a score - it is to answer the tuning questions that the
arithmetic could not:

  * How many steps do instances really take?
  * How fast does context actually grow per step? The worst-case estimate
    (~2900 tok/step, giving ~11 steps in a 32K window) assumes every observation
    hits the 10K-char cap. Typical bash output is far shorter, so the real figure
    could be several times better. This measures it.
  * Do runs die from context overflow, the step limit, or format errors? Those
    demand opposite fixes, and they are indistinguishable from a bare score.

Reads the trajectories mini-swe-agent writes at
<out>/<instance_id>/<instance_id>.traj.json (structure read from
minisweagent/run/benchmarks/swebench.py:163-176).
"""

from __future__ import annotations

import json
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "swebench_pilot")

trajs = sorted(out.glob("*/*.traj.json"))
if not trajs:
    print(f"!! no trajectories under {out}")
    raise SystemExit(1)

print(f"=== SWE-bench pilot: {len(trajs)} trajectories under {out} ===\n")

CHARS_PER_TOK = 3.6  # rough for code+English; only used for sizing, never quoted

exits: Counter[str] = Counter()
steps_all: list[int] = []
ctx_all: list[float] = []
rows = []

for t in trajs:
    try:
        d = json.loads(t.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  !! unreadable {t.name}: {e}")
        continue

    info = d.get("info", {})
    status = info.get("exit_status") or "unknown"
    exits[status] += 1

    msgs = d.get("messages") or d.get("trajectory") or []
    # one step = one assistant turn
    assistant = [m for m in msgs if m.get("role") == "assistant"]
    steps = len(assistant)
    total_chars = sum(len(str(m.get("content") or "")) for m in msgs)
    ctx_tok = total_chars / CHARS_PER_TOK

    submission = (info.get("submission") or "").strip()
    steps_all.append(steps)
    ctx_all.append(ctx_tok)
    rows.append((t.parent.name, status, steps, ctx_tok, len(submission)))

print(f"{'instance':<34} {'exit_status':<26} {'steps':>5} {'~ctx tok':>9} {'patch ch':>9}")
for name, status, steps, ctx, patch in sorted(rows, key=lambda r: -r[3]):
    print(f"{name:<34} {status:<26} {steps:>5} {ctx:>9,.0f} {patch:>9,}")

print("\n=== exit statuses (this is the tuning signal) ===")
for s, c in exits.most_common():
    note = ""
    low = s.lower()
    if "context" in low:
        note = "  <-- context window too small: shrink observations or step_limit"
    elif "limit" in low or "step" in low:
        note = "  <-- hit step_limit: raise it if context allows"
    elif "format" in low:
        note = "  <-- format failures: the fence or reasoning parser is misbehaving"
    elif "submit" in low:
        note = "  <-- healthy completion"
    print(f"  {s:<30} {c:>3}{note}")

if steps_all:
    print("\n=== step and context growth ===")
    print(f"  steps      median {stats.median(steps_all):.0f}   range [{min(steps_all)}, {max(steps_all)}]")
    print(f"  ~ctx tok   median {stats.median(ctx_all):,.0f}   max {max(ctx_all):,.0f}   (window is 32,768)")
    nz = [c / s for c, s in zip(ctx_all, steps_all) if s]
    if nz:
        per = stats.median(nz)
        print(f"  ~tok/step  median {per:,.0f}")
        print(f"  -> a 32,768 window supports roughly {32768 / per:.0f} steps at this rate")
        print("     (worst-case arithmetic said ~11; if this is much higher, raise step_limit)")

empty = [r for r in rows if r[4] == 0]
print(f"\n=== patches ===\n  empty submissions: {len(empty)}/{len(rows)}")
if empty:
    print("  empty means the agent never reached the submit command - usually the")
    print("  step limit or a context overflow, not a wrong fix. Check exit_status above.")
