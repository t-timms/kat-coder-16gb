#!/usr/bin/env bash
# Grade the pilot's 4 real patches with the official SWE-bench harness.
#
# This is the last unproven link in the chain. Generating 50 instances takes ~3.4
# hours; discovering afterwards that grading does not run would waste all of it.
# Proving it on 5 costs minutes.
#
# The harness wants predictions as JSONL with instance_id / model_patch /
# model_name_or_path. mini-swe-agent writes preds.json as a dict keyed by
# instance_id, so it is converted here.
set -uo pipefail

OUT="${1:-$HOME/swebench_pilot}"
RUN_ID="kat_pilot_$(date +%H%M%S)"
ENVBIN=~/swebench-env/bin

echo "=== converting preds.json -> preds.jsonl ==="
"$ENVBIN/python" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
d = json.loads((out / "preds.json").read_text())
rows, empty = [], 0
for iid, v in d.items():
    patch = (v.get("model_patch") or "").strip()
    if not patch:
        empty += 1
        continue
    rows.append({
        "instance_id": iid,
        "model_patch": v["model_patch"],
        "model_name_or_path": v.get("model_name_or_path") or "kat-16gb",
    })
p = out / "preds.jsonl"
p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
print(f"   {len(rows)} gradable, {empty} empty (empty ones cannot resolve and are")
print("   correctly counted as unresolved in the denominator later)")
PY

echo
echo "=== running the official harness (images already cached locally) ==="
cd "$OUT" || exit 1
"$ENVBIN/python" -m swebench.harness.run_evaluation \
  --dataset_name SWE-bench/SWE-bench_Verified \
  --predictions_path "$OUT/preds.jsonl" \
  --max_workers 2 \
  --run_id "$RUN_ID" 2>&1 | tail -30

echo
echo "=== ARTIFACT: the report file, not the exit code ==="
report=$(ls -1t "$OUT"/*.json 2>/dev/null | grep -i "$RUN_ID" | head -1)
if [ -z "$report" ]; then
  report=$(ls -1t ./*."$RUN_ID".json 2>/dev/null | head -1)
fi
if [ -n "$report" ] && [ -f "$report" ]; then
  echo "   $report"
  "$ENVBIN/python" - "$report" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("total_instances", "submitted_instances", "completed_instances",
          "resolved_instances", "unresolved_instances", "empty_patch_instances",
          "error_instances"):
    if k in d:
        print(f"   {k:<24}: {d[k]}")
PY
else
  echo "   !! no report json found; listing what was written:"
  ls -la "$OUT" | tail -10
fi
