#!/usr/bin/env bash
# Timed pilot: measure throughput before committing to a long paired run.
#
# Purpose is a NUMBER, not a result. It answers: how long does one document cost
# on a 36 GB bf16 checkpoint streamed over PCIe 4.0 x16 against 16.3 GB of VRAM?
# Everything downstream (how many documents we can afford, and therefore what
# effect size the experiment can resolve) depends on this one measurement.
#
# Constraints that are not negotiable on this machine:
#   * enforce_eager=True - CUDA graph capture is numerically broken on SM120
#   * no chat template - loglikelihood scoring, wrapping it changes what is scored
#   * log_samples - the real comparison is PAIRED, needs per-document values
#   * full log to file, never a filtered pipe, so a failure keeps its cause

set -uo pipefail

N="${1:-32}"
SEED_ARM="${2:-42}"

ROOT="${HOME}/reap-stability"
LM="${HOME}/vllm-env/bin/lm_eval"
OUT="${HOME}/reap-eval/pilot_n${N}_seed${SEED_ARM}"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

ckpt=$(find "${ROOT}/n64_s${SEED_ARM}" -type d -name "reap-*-0.50" 2>/dev/null | head -1)
if [ -z "${ckpt}" ]; then
  echo "!! no 0.50 checkpoint for seed ${SEED_ARM}"
  exit 1
fi

mkdir -p "${OUT}"
echo "checkpoint : ${ckpt}"
echo "documents  : ${N}"
echo "started    : $(date -Iseconds)"

start=$(date +%s)

"${LM}" run \
  --model vllm \
  --model_args "pretrained=${ckpt},dtype=bfloat16,max_model_len=4096,enforce_eager=True,cpu_offload_gb=40,gpu_memory_utilization=0.85,trust_remote_code=True" \
  --tasks magicoder_code_ppl \
  --include_path "${HOME}/lm_eval_tasks" \
  --batch_size auto \
  --limit "${N}" \
  --log_samples \
  --output_path "${OUT}" \
  --seed 1234 \
  > "${OUT}/pilot.log" 2>&1

rc=$?
end=$(date +%s)
elapsed=$((end - start))

echo "finished   : $(date -Iseconds)"
echo "rc         : ${rc}  (not trusted on this stack, artifacts below decide)"
echo "elapsed    : ${elapsed}s"

# Judge by artifact and by a parsed metric, never by exit code.
res=$(find "${OUT}" -name 'results_*.json' 2>/dev/null | head -1)
samp=$(find "${OUT}" -name '*.jsonl' 2>/dev/null | head -1)

if [ -n "${res}" ]; then
  echo "results    : ${res}"
  grep -oE '"(word_perplexity|byte_perplexity|bits_per_byte),none": *[0-9.]+' "${res}" || echo "  (no metric parsed)"
else
  echo "!! NO results json - eval did not produce output"
fi

if [ -n "${samp}" ]; then
  n_lines=$(wc -l < "${samp}")
  echo "samples    : ${samp} (${n_lines} per-document records)"
else
  echo "!! NO per-document samples - the paired test would be impossible"
fi

if [ "${elapsed}" -gt 0 ] && [ -n "${samp}" ]; then
  echo
  echo "=== extrapolation ==="
  awk -v e="${elapsed}" -v n="${N}" 'BEGIN {
    per = e / n
    printf "  %.2f s per document\n", per
    printf "  1000 docs  -> %.1f min per arm, %.1f min for both\n", per*1000/60, per*2000/60
    printf "  4000 docs  -> %.1f min per arm, %.1f min for both\n", per*4000/60, per*8000/60
    printf "  10000 docs -> %.1f h per arm,  %.1f h for both\n",  per*10000/3600, per*20000/3600
  }'
fi

echo
echo "last 12 log lines:"
tail -12 "${OUT}/pilot.log"
