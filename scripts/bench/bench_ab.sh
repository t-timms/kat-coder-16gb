#!/usr/bin/env bash
# A/B latency benchmark: NVFP4A16 (Marlin dequant) vs NVFP4 W4A4 (native FP4).
#
# Uses `vllm bench latency`, the tool that ships with vLLM, NOT a hand-rolled
# script. A bespoke throughput script once reported 6.5 tok/s for a workload the
# official tool measured at 111.3 - wrong by 17x.
#
# MEASUREMENT RULES BEING FOLLOWED
#   * >=5 SEPARATE PROCESS INVOCATIONS per arm. Within-run spread is not evidence:
#     batch-1 decode on this machine once measured 9.6 / 32.2 / 23.7 tok/s across
#     three runs of an identical command, each internally tight. Variance lives
#     BETWEEN invocations.
#   * INTERLEAVED A,B,A,B - never blocked. Blocked runs let thermal drift and run
#     order masquerade as the effect.
#   * Warm cache. The first invocation of each arm is discarded as cache-warming,
#     because a cold JIT/compile cache measures the compiler, not the model.
#   * Report MEDIAN AND RANGE, never a single number.
#
# batch-size 1 because an interactive coding agent is a single-stream workload.

set -uo pipefail

REPS="${1:-5}"
A16="${HOME}/models/kat-50pct-nvfp4a16"
W4A4="${HOME}/models/kat-50pct-nvfp4-w4a4"
OUT="${HOME}/bench-ab"
BIN="${HOME}/vllm-env/bin/vllm"

INPUT_LEN=512
OUTPUT_LEN=256

mkdir -p "${OUT}"
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

run_one() {
  local tag="$1" model="$2" rep="$3"
  local json="${OUT}/${tag}_rep${rep}.json"
  local log="${OUT}/${tag}_rep${rep}.log"

  if [ ! -d "${model}" ]; then
    echo "    skip ${tag}: ${model} does not exist"
    return
  fi

  "${BIN}" bench latency \
    --model "${model}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --batch-size 1 \
    --num-iters-warmup 1 \
    --num-iters 3 \
    --output-json "${json}" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --enforce-eager \
    --language-model-only \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    > "${log}" 2>&1

  local rc=$?
  if [ -f "${json}" ]; then
    echo "    ${tag} rep${rep} OK  $(grep -oE '"avg_latency":[0-9.]+' "${json}" | head -1)"
  else
    echo "    ${tag} rep${rep} !! NO JSON (rc=${rc})"
    grep -aiE "error|assert" "${log}" | tail -3
  fi
}

echo "=== A/B latency, ${REPS} reps each, interleaved, started $(date -Iseconds) ==="
echo "    A = NVFP4A16 (weight-only, Marlin)"
echo "    B = NVFP4 W4A4 (native FP4, VLLM_CUTLASS)"
echo "    in=${INPUT_LEN} out=${OUTPUT_LEN} batch=1"
echo
echo "--- rep 0 (WARMUP, discarded: a cold compile cache measures the compiler) ---"
run_one a16 "${A16}" 0
run_one w4a4 "${W4A4}" 0

for rep in $(seq 1 "${REPS}"); do
  echo "--- rep ${rep} @ $(date -Iseconds) ---"
  run_one a16 "${A16}" "${rep}"
  run_one w4a4 "${W4A4}" "${rep}"
done

echo
echo "=== which kernel did each arm actually select? ==="
echo "  (a silent fallback shows up as throughput loss with no error)"
grep -h "NvFp4 MoE backend" "${OUT}"/a16_rep1.log 2>/dev/null | tail -1
grep -h "NvFp4 MoE backend" "${OUT}"/w4a4_rep1.log 2>/dev/null | tail -1

echo
echo "=== done $(date -Iseconds) ==="
echo "run ~/bench_ab_analyze.py for median and range"
