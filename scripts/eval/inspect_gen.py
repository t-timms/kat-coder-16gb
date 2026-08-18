"""Are the generations truncated, or complete?

A pass@1 of 1.0 on ten problems says nothing about whether the harness is set up
correctly, because the first ten HumanEval problems are trivial and the benchmark is
contaminated. What the pilot is actually for is checking two failure modes that would
depress a full run for formatting reasons:

  1. Stop sequences firing inside the reasoning prose, before any code appears.
     humaneval_instruct inherits until: ["\\nclass","\\ndef","\\n#","\\nif","\\nprint"].
  2. Hitting max_gen_toks (1024) mid-thought, as happened in the smoke test at 256.

Both look identical in the score. Only the text distinguishes them.
"""

from __future__ import annotations

import json
from pathlib import Path

root = Path.home() / "eval-pilot"
files = sorted(root.rglob("samples_*.jsonl"))
if not files:
    raise SystemExit(f"no samples under {root}")

lines = files[0].read_text().splitlines()
print(f"records: {len(lines)}\n")

lengths = []
suspicious = []

for i, line in enumerate(lines):
    r = json.loads(line)
    resps = r.get("filtered_resps") or r.get("resps") or []
    text = resps[0] if isinstance(resps, list) and resps else ""
    if isinstance(text, list):
        text = text[0] if text else ""
    text = str(text)
    lengths.append(len(text))

    # A complete answer should contain a def and a return, and should not end
    # mid-token or mid-line.
    looks_complete = ("def " in text) and ("return" in text or "yield" in text)
    ends_abruptly = bool(text) and not text.rstrip().endswith(("`", ")", ":", "]", "}", "\"", "'", ".", "e"))
    if not looks_complete or ends_abruptly:
        suspicious.append((i, len(text), text[-120:] if text else "(empty)"))

print(f"response chars: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
print(f"suspicious (incomplete-looking): {len(suspicious)}/{len(lines)}\n")

for i, n, tail in suspicious[:4]:
    print(f"  doc {i} ({n} chars) ends: ...{tail!r}\n")

print("=== full text of the first response ===")
r0 = json.loads(lines[0])
resp = r0.get("filtered_resps") or r0.get("resps")
if isinstance(resp, list):
    resp = resp[0]
if isinstance(resp, list):
    resp = resp[0]
print(str(resp)[:1200])
