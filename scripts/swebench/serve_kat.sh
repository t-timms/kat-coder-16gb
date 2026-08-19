#!/usr/bin/env bash
# Optimal vLLM serve script for KAT-Coder-16gb
# Updated: 2026-08-19 with tested optimal config
#
# Key optimizations:
# - max_num_seqs=8 (4x higher than previous max_seqs=2, tested optimal)
# - cudagraph_capture_sizes=[1,2] (minimal capture, fastest startup)
# - cudagraph_mode=PIECEWISE (required for hybrid attention)
# - FP8 KV cache (50% memory savings)
# - Prefix caching (reduces prompt reprocessing)
# - Tool calling with qwen3_xml parser (required for mini-swe-agent)
#
# Performance metrics (tested 2026-08-19):
# - Concurrency: 1.86x for 32K context
# - CUDA graph memory: 0.04 GiB (minimal overhead)
# - Startup time: ~40 seconds
set -uo pipefail

MODEL=~/models/kat-50pct-nvfp4a16-renorm-stripped
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-32768}"
LOG=~/kat_serve.log

echo "=== starting vLLM (full log -> $LOG) ==="
setsid ~/vllm-env/bin/vllm serve "$MODEL" \
  --served-model-name kat-16gb \
  --port "$PORT" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --language-model-only \
  > "$LOG" 2>&1 &

echo $! > ~/kat_serve.pid
echo "pid $(cat ~/kat_serve.pid), waiting for readiness"

for i in $(seq 1 150); do
  if ! kill -0 "$(cat ~/kat_serve.pid)" 2>/dev/null; then
    echo "!! died. innermost error:"
    grep -aE "ValueError|RuntimeError|Error:" "$LOG" | tail -5
    exit 1
  fi
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200; then
    echo "READY after ~$((i*4))s"
    grep -a "GPU KV cache size\|Maximum concurrency\|CUDA graph" "$LOG" | tail -5
    exit 0
  fi
  sleep 4
done

echo "!! timed out waiting for readiness"
tail -20 "$LOG"
exit 1
