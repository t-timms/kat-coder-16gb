#!/usr/bin/env bash
# Every precondition for quantizing the pruned checkpoint, in ONE pass.
#
# Discovering these serially costs a failed multi-hour run each time. The plan is
# to quantize the existing 50%-pruned bf16 checkpoint (36 GB) to NVFP4 (~11-12 GB)
# so it fits 16.3 GB natively, which sidesteps the CPU-offload UVA crash entirely
# and finally exercises the NVFP4 kernels on SM120 for qwen3_5_moe.

set -uo pipefail

echo "=== 1. which env has llm-compressor / compressed-tensors ? ==="
for e in quant-env vllm-env reap-env reap-cuda-env mlenv; do
  sp="${HOME}/${e}/lib/python3.12/site-packages"
  if [ -d "${sp}" ]; then
    lc=$(ls "${sp}" | grep -ciE '^llmcompressor' || true)
    ct=$(ls "${sp}" | grep -ciE '^compressed_tensors' || true)
    tf=$(cat "${sp}"/transformers/__init__.py 2>/dev/null | grep -m1 '^__version__' | cut -d'"' -f2)
    printf '  %-14s llmcompressor=%s compressed_tensors=%s transformers=%s\n' \
      "${e}" "${lc}" "${ct}" "${tf:-none}"
  else
    printf '  %-14s (no site-packages)\n' "${e}"
  fi
done

echo
echo "=== 2. llm-compressor version, if any ==="
for e in quant-env vllm-env reap-env; do
  d=$(ls -d "${HOME}/${e}"/lib/python3.12/site-packages/llmcompressor-*.dist-info 2>/dev/null | head -1)
  if [ -n "${d}" ]; then
    printf '  %-12s %s\n' "${e}" "$(basename "${d}")"
  fi
done

echo
echo "=== 3. does llm-compressor know REAP? (would let us prune+quantize in one pass) ==="
for e in quant-env vllm-env reap-env; do
  p="${HOME}/${e}/lib/python3.12/site-packages/llmcompressor/modifiers/pruning"
  if [ -d "${p}" ]; then
    printf '  %-12s pruning modifiers: %s\n' "${e}" "$(ls "${p}" | tr '\n' ' ')"
  fi
done

echo
echo "=== 4. the pruned checkpoints we would quantize ==="
find "${HOME}/reap-stability" -type d -name 'reap-*-0.50' -exec du -sh {} +

echo
echo "=== 5. transformers requirement: qwen3_5 needs >= 5.2.0 ==="
echo "  (vllm-env transformers is 4.57.1 and CANNOT load qwen3_5_moe;"
echo "   reap-env has 5.15.0. Quantization must run where transformers can load it.)"

echo
echo "=== 6. disk headroom (vhdx grows 1:1 now, no slack left) ==="
df -h / | tail -1
echo "  a NVFP4 output is ~11-12 GB"

echo
echo "=== 7. GPU free? ==="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
