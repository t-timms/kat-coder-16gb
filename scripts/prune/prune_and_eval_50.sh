#!/usr/bin/env bash
# The decisive experiment at 50% sparsity, which is where it can actually work.
#
# Why 50% and not 25%: REAP reports 0.16% mean accuracy loss at 25% sparsity and
# 1.2% at 50%. At 0.16% the baseline damage is so small that two differently
# calibrated prunes cannot be told apart at any item count, so the 25% version of
# this experiment is underpowered by construction.
#
# Measured from the cached n=64 observations, the two calibration draws disagree
# on ~8-9 experts PER LAYER at every rate. What changes with rate is where those
# experts sit: at 25% they are all among the least salient and safe to cut either
# way; at 50% the cut reaches experts that matter. So 50% is a more SENSITIVE
# test, not merely a harsher one.
#
# Reuses the same cached observations (same --artifacts-dir), so no
# recalibration. Only --compression-ratio and --seed vary.

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/KAT-Coder-V2.5-Dev"
ROOT="${HOME}/reap-stability"
RATIO=0.50

echo "waiting for the 25% prunes to finish..."
while pgrep -f 'prune_both_seeds.sh' > /dev/null; do
  sleep 60
done
echo "clear at $(date -Iseconds)"

for seed in 42 0; do
  rundir="${ROOT}/n64_s${seed}"
  log="${rundir}/prune50.log"

  existing=$(find "${rundir}" -type d -name '*-0.5' -o -type d -name '*-0.50' 2>/dev/null | head -1)
  if [ -n "${existing}" ] && [ -n "$(find "${existing}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
    echo "skip seed ${seed} at 50%: already exists"
    continue
  fi

  echo "=== pruning seed ${seed} at ${RATIO} @ $(date -Iseconds) ==="

  "${BIN}" prune layerwise \
    --model "${MODEL}" \
    --dataset theblackcat102/evol-codealpaca-v1 \
    --compression-ratio "${RATIO}" \
    --prune-method reap \
    --observe-backend bmm \
    --residency cpu_full \
    --batch-size 1 \
    --batches-per-category 64 \
    --model-max-length 2048 \
    --seed "${seed}" \
    --artifacts-dir "${rundir}" \
    --no-eval \
    > "${log}" 2>&1

  rc=$?

  if grep -q "Aggregate cache hit" "${log}"; then
    echo "    observation cache REUSED (no recalibration)"
  else
    echo "    !! WARNING: no cache hit, may have recalibrated - comparison invalid"
  fi

  out=$(find "${rundir}" -type d -name '*-0.5' -o -type d -name '*-0.50' 2>/dev/null | head -1)
  if [ -n "${out}" ] && [ -n "$(find "${out}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
    echo "    seed ${seed} rc=${rc} checkpoint OK"
    du -sh "${out}"
    # Confirm the config actually says 128 experts, not just that files exist.
    python3 - "${out}" <<'PY'
import json, sys, pathlib
cfg = json.loads((pathlib.Path(sys.argv[1]) / "config.json").read_text())
tc = cfg.get("text_config", {})
print(f"    text_config.num_experts = {tc.get('num_experts')} (expect 128)")
PY
  else
    echo "    seed ${seed} rc=${rc} !! NO CHECKPOINT - see ${log}"
  fi
done

echo "=== 50% prunes complete $(date -Iseconds) ==="
