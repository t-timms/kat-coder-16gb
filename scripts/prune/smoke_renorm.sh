#!/usr/bin/env bash
# Prove the renormalization fix actually fires, before spending ~6 h on six seeds.
#
# The patch makes the adapter report True. This checks the value survives all the
# way into the observer config at runtime, by looking for the log line pipeline.py
# emits only when renormalization is ON:
#     "Renormalizing topk router weights to sum to 1."
#
# FRESH artifacts dir is mandatory: every cached observations_*.pt was computed
# WITHOUT renormalization, and reap caches by path, so reusing a dir would return
# a stale hit and this check would prove nothing.
#
# One sample only. We are testing a code path, not measuring anything.

set -uo pipefail

BIN="${HOME}/reap-cuda-env/bin/reap"
MODEL="${HOME}/models/KAT-Coder-V2.5-Dev"
DIR="${HOME}/reap-renorm-smoke"
LOG="${DIR}/smoke.log"

rm -rf "${DIR}"
mkdir -p "${DIR}"

echo "started $(date -Iseconds)"

"${BIN}" prune layerwise \
  --model "${MODEL}" \
  --dataset theblackcat102/evol-codealpaca-v1 \
  --compression-ratio 0.25 \
  --prune-method reap \
  --observe-backend bmm \
  --residency cpu_full \
  --batch-size 1 \
  --batches-per-category 1 \
  --model-max-length 2048 \
  --seed 42 \
  --artifacts-dir "${DIR}" \
  --observe-only \
  --no-eval \
  > "${LOG}" 2>&1

rc=$?
echo "rc=${rc} (not trusted) finished $(date -Iseconds)"

echo
echo "=== the decisive line ==="
if grep -q "Renormalizing topk router weights to sum to 1" "${LOG}"; then
  echo "  PASS: renormalization is ON"
  grep -n "Renormalizing topk router weights" "${LOG}"
else
  echo "  FAIL: renormalization did NOT engage"
  echo "  (searching for any renorm mention)"
  grep -in "renorm" "${LOG}" | head -5
fi

echo
echo "=== cache must be freshly computed, not a hit ==="
if grep -q "Aggregate cache hit" "${LOG}"; then
  echo "  !! CACHE HIT - this run proved nothing, the dir was not fresh"
else
  echo "  OK: no cache hit, observations computed fresh"
fi

echo
echo "=== artifact written? ==="
find "${DIR}" -name 'observations_*.pt' -exec ls -lh {} +
