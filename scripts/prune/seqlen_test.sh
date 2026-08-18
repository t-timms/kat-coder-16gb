#!/usr/bin/env bash
# Given a FIXED token budget, is it better spent on many short samples or few
# long ones?
#
# This is the honest framing. At a fixed budget, sequence length and sample
# count are inversely locked, so no experiment can vary one alone. The earlier
# single-seed design implied it could, which was wrong.
#
# Runs BOTH seeds at 16 x 8192, which is what makes stability measurable here.
# With one seed we could only compare prune sets across configs; with two we get
# a seed-to-seed overlap directly comparable to the sweep's own numbers.
#
# The comparison set, all already on disk except this run:
#
#   config          samples  seqlen   tokens   role
#   n16_s42/s0           16    2048   32,768   same COUNT, fewer tokens
#   n64_s42/s0           64    2048  131,072   same TOKENS, more samples
#   L8192_s42/s0         16    8192  131,072   this run
#
# Two questions it answers:
#   1. Stability at equal tokens: is L8192 seed-overlap ~= n64 seed-overlap?
#      If yes, TOKENS drive convergence and shape is free.
#      If L8192 is clearly worse, SAMPLE DIVERSITY drives it, and short-and-many
#      wins, which also means truncating agentic trajectories is acceptable.
#   2. Whether length adds anything beyond tokens: L8192 vs n16 (same count).
#
# residency=cpu_full: the validated deterministic path. Never mix in layerwise,
# which disagrees with it on 18% of pruning decisions.

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/KAT-Coder-V2.5-Dev"
ROOT="${HOME}/reap-stability"
SEQLEN=8192
SAMPLES=16

echo "waiting for the stability sweep to finish..."
while pgrep -f 'stability_run.sh' > /dev/null; do
  sleep 60
done
echo "sweep clear at $(date -Iseconds)"

for seed in 42 0; do
  rundir="${ROOT}/L8192_s${seed}"
  log="${rundir}/run.log"

  if find "${rundir}" -name 'observations_*.pt' 2>/dev/null | grep -q .; then
    echo "skip seed ${seed} (already done)"
    continue
  fi

  mkdir -p "${rundir}"
  echo "=== ${SAMPLES} x ${SEQLEN} = $(( SAMPLES * SEQLEN )) tokens, seed ${seed} @ $(date -Iseconds) ==="

  "${BIN}" prune layerwise \
    --model "${MODEL}" \
    --dataset theblackcat102/evol-codealpaca-v1 \
    --compression-ratio 0.25 \
    --prune-method reap \
    --observe-backend bmm \
    --residency cpu_full \
    --batch-size 1 \
    --batches-per-category "${SAMPLES}" \
    --model-max-length "${SEQLEN}" \
    --seed "${seed}" \
    --artifacts-dir "${rundir}" \
    --observe-only \
    --no-eval \
    > "${log}" 2>&1

  rc=$?
  if find "${rundir}" -name 'observations_*.pt' 2>/dev/null | grep -q .; then
    echo "    seed ${seed} rc=${rc} artifact OK"
  else
    echo "    seed ${seed} rc=${rc} !! NO ARTIFACT - see ${log}"
  fi
done

echo "=== seqlen test complete $(date -Iseconds) ==="
