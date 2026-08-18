#!/usr/bin/env bash
# The decisive experiment: does the calibration DRAW change the pruned model?
#
# Two checkpoints identical in every respect except which 64 calibration samples
# were drawn (seed 42 vs seed 0). They disagree on ~12% of pruned experts. The
# pipeline is bit-for-bit deterministic and loglikelihood scoring is deterministic,
# so the noise floor is exactly zero: any difference is attributable to the draw.
#
# Reading A: ranking is unstable, pruning decisions are partly noise.
# Reading B: ranking is underdetermined but harmless, many prune sets are equal.
#
# Non-negotiable on this machine:
#   * `run` subcommand, NOT `eval` (that subcommand does not exist)
#   * enforce_eager=True - CUDA graph capture is numerically broken on SM120
#   * checkpoint glob pinned to -0.50 so a future re-prune cannot silently
#     substitute a different sparsity
#   * log_samples - the comparison is PAIRED and needs per-document values
#   * full unfiltered log per arm, so a failure keeps the line naming its cause

set -uo pipefail

N="${1:-1000}"
ROOT="${HOME}/reap-stability"
LM="${HOME}/vllm-env/bin/lm_eval"
OUT="${HOME}/reap-eval/paired_n${N}"

export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0

mkdir -p "${OUT}"
echo "=== paired run, ${N} documents per arm, started $(date -Iseconds) ==="

for seed in 42 0; do
  ckpt=$(find "${ROOT}/n64_s${seed}" -type d -name "reap-*-0.50" 2>/dev/null | head -1)
  if [ -z "${ckpt}" ]; then
    echo "!! seed ${seed}: no 0.50 checkpoint, ABORTING (a one-armed result is worthless)"
    exit 1
  fi

  outdir="${OUT}/seed${seed}"
  if [ -n "$(find "${outdir}" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
    echo "skip seed ${seed}: results already present"
    continue
  fi

  mkdir -p "${outdir}"
  echo
  echo "--- seed ${seed} @ $(date -Iseconds) ---"
  echo "    ${ckpt}"
  start=$(date +%s)

  "${LM}" run \
    --model vllm \
    --model_args "pretrained=${ckpt},dtype=bfloat16,max_model_len=4096,enforce_eager=True,cpu_offload_gb=40,gpu_memory_utilization=0.85,trust_remote_code=True" \
    --tasks magicoder_code_ppl \
    --include_path "${HOME}/lm_eval_tasks" \
    --batch_size auto \
    --limit "${N}" \
    --log_samples \
    --output_path "${outdir}" \
    --seed 1234 \
    > "${outdir}/eval.log" 2>&1

  rc=$?
  elapsed=$(( $(date +%s) - start ))

  res=$(find "${outdir}" -name 'results_*.json' 2>/dev/null | head -1)
  samp=$(find "${outdir}" -name 'samples_*.jsonl' 2>/dev/null | head -1)

  echo "    rc=${rc} (not trusted) elapsed=${elapsed}s"
  if [ -n "${res}" ]; then
    grep -oE '"(word_perplexity|byte_perplexity|bits_per_byte),none": *[0-9.]+' "${res}"
  else
    echo "    !! NO results json"
  fi
  if [ -n "${samp}" ]; then
    echo "    per-document records: $(wc -l < "${samp}")"
  else
    echo "    !! NO per-document samples - paired test impossible"
    tail -8 "${outdir}/eval.log"
  fi
done

echo
echo "=== paired run complete $(date -Iseconds) ==="
