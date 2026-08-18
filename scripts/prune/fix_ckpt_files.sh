#!/usr/bin/env bash
# Restore the processor/tokenizer files reap's save path drops.
#
# KAT-Coder is Qwen3_5MoeForConditionalGeneration, a multimodal shell, so the
# loader constructs an image processor and fails without preprocessor_config.json.
# Expert pruning touches only the MoE experts in the language model, so every file
# copied here is byte-identical to the base model's and safe to carry over.
#
# Same failure class as the ZAYA1 checkpoint that shipped without tokenizer files.

set -uo pipefail

BASE="${HOME}/models/KAT-Coder-V2.5-Dev"
NEEDED=(preprocessor_config.json video_preprocessor_config.json merges.txt vocab.json)

targets=$(find "${HOME}/reap-stability" -type d -name 'reap-*-0.50' 2>/dev/null)

if [ -z "${targets}" ]; then
  echo "!! no 0.50 checkpoints found"
  exit 1
fi

for ckpt in ${targets}; do
  echo "=== ${ckpt##*/} ==="
  for f in "${NEEDED[@]}"; do
    if [ -f "${ckpt}/${f}" ]; then
      echo "  present already: ${f}"
    elif [ -f "${BASE}/${f}" ]; then
      cp "${BASE}/${f}" "${ckpt}/${f}"
      if [ -f "${ckpt}/${f}" ]; then
        echo "  copied: ${f} ($(stat -c%s "${ckpt}/${f}") bytes)"
      else
        echo "  !! COPY FAILED: ${f}"
      fi
    else
      echo "  !! not in base either: ${f}"
    fi
  done
  echo "  file count now: $(ls -1 "${ckpt}" | wc -l)"
done

echo
echo "=== both checkpoints must now match on file inventory ==="
for ckpt in ${targets}; do
  echo "${ckpt##*/}: $(ls -1 "${ckpt}" | sort | tr '\n' ' ')"
done
