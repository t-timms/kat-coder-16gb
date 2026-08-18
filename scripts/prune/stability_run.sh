#!/usr/bin/env bash
# Calibration size vs expert-ranking stability.
#
# The question: how much calibration does REAP need before the SET OF EXPERTS IT
# PRUNES stops moving? Below that point, no criteria comparison can distinguish
# a real effect from resampling noise.
#
# Method: the pipeline is bit-for-bit deterministic (verified: two cpu_full runs
# of the same config agreed 100.00%), so repetition tells us nothing. Stability
# has to be probed by RESAMPLING instead: run each size with two different seeds,
# which draws different calibration samples, and measure how much the pruning
# decision changes. Converging overlap as size grows means the ranking is
# settling.
#
# Sizes bracket the published plateau. A Mixtral 8x7B sweep found 64-128
# sequences of 2048 tokens best, and another study reports accuracy plateauing
# near 150K tokens. Sequence length is fixed at 2048 to match that convention,
# and because our own scaling measurement showed long sequences are cheaper per
# token (~8.07s fixed cost per batch + ~50 tok/s).
#
# residency=cpu_full deliberately: it is the validated, reproducible path.
# Layerwise disagrees with it on 18% of pruning decisions and must not be mixed
# into a comparison.

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/KAT-Coder-V2.5-Dev"
ROOT="${HOME}/reap-stability"
SEQLEN=2048

mkdir -p "${ROOT}"

# sequences: 4=8K tokens, 16=32K, 64=131K tokens (spans well below to within
# the published plateau)
SIZES=(4 16 64)
SEEDS=(42 0)

echo "=== calibration stability sweep start $(date -Iseconds) ==="

for size in "${SIZES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    tokens=$(( size * SEQLEN ))
    rundir="${ROOT}/n${size}_s${seed}"
    log="${rundir}/run.log"

    if find "${rundir}" -name 'observations_*.pt' 2>/dev/null | grep -q .; then
      echo "skip n=${size} seed=${seed} (done)"
      continue
    fi

    mkdir -p "${rundir}"
    echo "=== n=${size} seed=${seed}: ${tokens} tokens @ $(date -Iseconds) ==="

    "${BIN}" prune layerwise \
      --model "${MODEL}" \
      --dataset theblackcat102/evol-codealpaca-v1 \
      --compression-ratio 0.25 \
      --prune-method reap \
      --observe-backend bmm \
      --residency cpu_full \
      --batch-size 1 \
      --batches-per-category "${size}" \
      --model-max-length "${SEQLEN}" \
      --seed "${seed}" \
      --artifacts-dir "${rundir}" \
      --observe-only \
      --no-eval \
      > "${log}" 2>&1

    rc=$?
    if find "${rundir}" -name 'observations_*.pt' 2>/dev/null | grep -q .; then
      echo "    rc=${rc} artifact OK"
    else
      echo "    rc=${rc} !! NO ARTIFACT - see ${log}"
    fi
  done
done

echo "=== sweep complete $(date -Iseconds) ==="
