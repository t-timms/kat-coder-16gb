#!/usr/bin/env bash
# The decisive experiment, part 1: build two pruned checkpoints that differ ONLY
# in which calibration samples were drawn.
#
# seed 42 and seed 0 at n=64 (131,072 tokens) disagree on ~12% of which experts
# to remove. If the resulting models score the same, that disagreement is
# harmless and the ranking is merely underdetermined. If they score differently,
# calibration draw materially changes model quality, which is a real warning.
#
# Reuses the EXISTING observation artifacts rather than recalibrating: same
# --artifacts-dir and same config means reap hits its aggregate cache
# ("Aggregate cache hit @ ...") and goes straight to pruning. That saves ~55 min
# per seed.
#
# Everything else is held identical: same model, same dataset, same
# compression-ratio, same backend, same residency. Only --seed differs.

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/KAT-Coder-V2.5-Dev"
ROOT="${HOME}/reap-stability"

for seed in 42 0; do
  rundir="${ROOT}/n64_s${seed}"
  log="${rundir}/prune.log"

  pruned=$(find "${rundir}" -type d -name 'reap-*' 2>/dev/null | head -1)
  if [ -n "${pruned}" ] && [ -n "$(find "${pruned}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
    echo "skip seed ${seed}: pruned checkpoint already exists at ${pruned}"
    continue
  fi

  echo "=== pruning seed ${seed} @ $(date -Iseconds) ==="

  "${BIN}" prune layerwise \
    --model "${MODEL}" \
    --dataset theblackcat102/evol-codealpaca-v1 \
    --compression-ratio 0.25 \
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

  # Confirm the observation cache was reused rather than silently recalibrated.
  if grep -q "Aggregate cache hit" "${log}"; then
    echo "    observation cache REUSED (no recalibration)"
  else
    echo "    !! WARNING: no cache hit, it may have recalibrated"
  fi

  # Judge by artifact, never exit code.
  out=$(find "${rundir}" -type d -name 'reap-*' 2>/dev/null | head -1)
  if [ -n "${out}" ] && [ -n "$(find "${out}" -name '*.safetensors' 2>/dev/null | head -1)" ]; then
    echo "    seed ${seed} rc=${rc} checkpoint OK: ${out}"
    du -sh "${out}"
  else
    echo "    seed ${seed} rc=${rc} !! NO CHECKPOINT - see ${log}"
  fi
done

echo "=== both prunes complete $(date -Iseconds) ==="
