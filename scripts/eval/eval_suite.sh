#!/usr/bin/env bash
# The accuracy suite for the release candidate.
#
# Three tasks, all instruct-framed because this is a chat model:
#   humaneval_instruct       164 problems, original tests. Contaminated and
#                            saturated, so reported for comparability only.
#   humaneval_plus_instruct  164 problems, EvalPlus's much stricter tests. Harder to
#                            pass by memorisation, so this is the meaningful number.
#   mbpp_plus_instruct       378 problems, EvalPlus MBPP.
#
# Every score is pass@1 with greedy decoding (do_sample: false, repeats: 1), so these
# are deterministic and re-runnable rather than sampled estimates.
#
# HumanEval executes model-generated code. Both gates are required: the
# --confirm_run_unsafe_code flag AND HF_ALLOW_CODE_EVAL=1.

set -uo pipefail

export HF_ALLOW_CODE_EVAL=1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LM="${HOME}/vllm-env/bin/lm_eval"
MODEL="${MODEL:-${HOME}/models/kat-50pct-nvfp4a16-renorm-stripped}"
TASKS_DIR="${REPO}/tasks"
OUT="${OUT:-${HOME}/eval-suite}"
LIMIT="${1:-}"

mkdir -p "${OUT}"

# lm-eval ships humaneval_instruct (instruct framing, original tests) and
# humaneval_plus (EvalPlus tests, completion framing) but not the combination,
# which is the number this model is reported on. tasks/humaneval_plus_instruct.yaml
# composes the two. Its `include:` and `!function utils....` references resolve
# relative to the yaml's own directory, so it has to be installed beside lm-eval's
# humaneval tasks; --include_path is not enough.
HE_DIR="$("${HOME}/vllm-env/bin/python" -c 'import lm_eval, os; print(os.path.join(os.path.dirname(lm_eval.__file__), "tasks", "humaneval"))')"
cp "${TASKS_DIR}/humaneval_plus_instruct.yaml" "${HE_DIR}/humaneval_plus_instruct.yaml"
echo "installed humaneval_plus_instruct -> ${HE_DIR}"

run_task() {
  local task="$1"
  local dir="${OUT}/${task}"
  if [ -n "$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
    echo "  skip ${task}: results already present"
    return 0
  fi
  mkdir -p "${dir}"

  local extra=()
  [ -n "${LIMIT}" ] && extra=(--limit "${LIMIT}")

  echo
  echo "--- ${task} @ $(date -Iseconds) ---"
  local start
  start=$(date +%s)

  "${LM}" run \
    --model vllm \
    --model_args "pretrained=${MODEL},dtype=bfloat16,max_model_len=2048,gpu_memory_utilization=0.92,max_num_seqs=4,language_model_only=True,trust_remote_code=True" \
    --tasks "${task}" \
    --include_path "${TASKS_DIR}" \
    --batch_size auto \
    "${extra[@]}" \
    --log_samples \
    --output_path "${dir}" \
    --apply_chat_template \
    --confirm_run_unsafe_code \
    --seed 1234 \
    > "${dir}/eval.log" 2>&1

  local rc=$? elapsed=$(( $(date +%s) - start ))
  local res
  res=$(find "${dir}" -name 'results_*.json' 2>/dev/null | head -1)

  if [ -n "${res}" ]; then
    # lm-eval spells this metric two ways depending on the task's filter:
    # humaneval reports "pass@1,create_test", mbpp reports "pass_at_1,extract_code".
    # Matching only the first silently produced an empty score for MBPP+ with rc=0.
    local score
    score=$(grep -oE '"pass(@|_at_)1,[a-z_]*": *[0-9.]+' "${res}" | head -1 | grep -oE '[0-9.]+$')
    [ -z "${score}" ] && score="!! score not found in ${res##*/}"
    local n
    n=$(find "${dir}" -name 'samples_*.jsonl' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
    printf '    pass@1 = %s   (n=%s, %ds, rc=%d)\n' "${score}" "${n}" "${elapsed}" "${rc}"
  else
    echo "    !! NO RESULTS (rc=${rc}, ${elapsed}s)"
    grep -aoE "ValueError: [^\"]{0,140}|Error: [^\"]{0,120}" "${dir}/eval.log" \
      | grep -v "Engine core init" | head -2
  fi
}

echo "=== accuracy suite, $(date -Iseconds) ==="
echo "    model: ${MODEL}"
[ -n "${LIMIT}" ] && echo "    LIMIT=${LIMIT} (pilot mode, not a real score)"

run_task humaneval_instruct
run_task humaneval_plus_instruct
run_task mbpp_plus_instruct

echo
echo "=== summary ==="
for t in humaneval_instruct humaneval_plus_instruct mbpp_plus_instruct; do
  r=$(find "${OUT}/${t}" -name 'results_*.json' 2>/dev/null | head -1)
  if [ -n "${r}" ]; then
    s=$(grep -oE '"pass@1,[a-z_]*": *[0-9.]+' "${r}" | head -1 | grep -oE '[0-9.]+$')
    printf '  %-26s %s\n' "${t}" "${s}"
  else
    printf '  %-26s (no result)\n' "${t}"
  fi
done
echo "=== done $(date -Iseconds) ==="
