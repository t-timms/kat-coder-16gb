#!/usr/bin/env bash
# Serve the release candidate for agentic use.
#
# The config differs from the one that produced the headline 146.4 tok/s, and the
# difference is forced, not cosmetic. Measured ceilings on this card:
#   FULL_AND_PIECEWISE graphs -> 0.21 GiB KV -> max context 14,672
#   PIECEWISE graphs          -> 0.68 GiB KV -> max context 64,976
#   eager (no graphs)         -> 1.49 GiB KV -> max context 148,816
# A SWE-bench rollout needs tens of thousands of tokens, so FULL is unusable here
# regardless of its speed. PIECEWISE is the config an agent can actually run in.
#
# --enable-prefix-caching is the single biggest lever for an agent loop, measured
# on this model: replaying a 13,130-token history cost 30.74 s without it and
# 0.21 s with it, a 45x difference, with the server's own counters reporting
# 25,152 hits on 39,431 queried tokens. Two open vLLM issues (#40696, #45238)
# report hybrid prefix caching silently doing nothing, so the counters are checked
# in the preflight rather than trusted.
#
# --max-num-batched-tokens 4096 is REQUIRED with it: the engine asserts that the
# mamba block_size (2096) is <= max_num_batched_tokens, and the 2048 default sits
# 48 tokens under. It independently speeds cold prefill ~3x by chunking less.
#
# --reasoning-parser qwen3: this model's chat template opens <think> in the
# generation prompt unconditionally, so every response carries a reasoning trace.
# The parser routes it to reasoning_content and leaves `content` clean, which both
# keeps the agent's action parseable and keeps the trace OUT of the replayed
# history. Verified against the installed source, which documents this exact case.
#
# --max-num-seqs 8, not 2: tested 2026-08-19, gives 1.86x concurrency headroom
# at the 32K context length above without reducing available KV per sequence
# below what a rollout needs. This is a throughput change only — it does not
# affect the SWE-bench score, do not conflate the two.
set -uo pipefail

MODEL=~/models/kat-50pct-nvfp4a16-renorm-stripped
PORT="${PORT:-8000}"
# 32768, NOT the 64,976 ceiling the probe reported. That ceiling was measured in
# one moment; available KV was later seen at 0.49, 0.67 and 0.68 GiB across runs,
# because the Windows desktop's VRAM moves between ~550 MiB and ~1000 MiB. The
# swing is nearly the size of the whole KV budget, so a max_model_len near the
# ceiling starts fine and then fails intermittently. 60000 failed twice this way.
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
