#!/usr/bin/env bash
# Serve + rollout + teardown in ONE process.
#
# Why combined: starting the server from a separate `wsl -- bash -lc` invocation
# reports READY, answers /v1/models, and then dies the moment that invocation
# exits, taking the process group with it.
#
# Two failures this script exists to prevent recurring:
#  1. Every LM call 400'd for the whole run with
#     `"auto" tool choice requires --enable-auto-tool-choice and
#     --tool-call-parser to be set`, because mini-swe-agent's default model class
#     always sends tools. Both flags are now set, and a preflight goes THROUGH
#     LITELLM with the same bash tool the agent uses - a curl preflight passed
#     while the real path was broken, because curl skipped the layer with the bug.
#  2. The rollout ran ~5 minutes on retry backoff before anyone noticed. The
#     preflight now fails the script before a single container is started.
set -uo pipefail

N="${1:-5}"
OUT="${2:-$HOME/swebench_pilot}"
PORT=8000
MODEL=~/models/kat-50pct-nvfp4a16-renorm-stripped
ENVBIN=~/swebench-env/bin
CFGDIR=~/kat_swebench
# Default is now the SOTA config, validated 2026-08-22 on a full 50-instance
# run: 26/50 = 52.0%, up from 40.0% at the old default (0 infrastructure
# failures, 0 crashes). This default reproduces that exact result from a
# fresh clone - kat_overrides_sota.yaml is kept unchanged from the run that
# produced it, deliberately. Set KAT_CONFIG=kat_overrides.yaml + MAXLEN=32768
# MAXSEQS=8 to reproduce the original 32K/40.0% baseline instead. Set
# KAT_CONFIG=kat_overrides_context_managed.yaml to try the still-unvalidated
# context-*budget* experiment (reduces max_tokens instead of raising the
# ceiling - a different, untested lever - see that file for what it changes).
# Set KAT_CONFIG=kat_overrides_sota_presence_penalty.yaml to try a candidate
# sampling-parameter fix, tested on one instance only, NOT full-pilot
# validated - see that file's header before citing any result from it.
KAT_CONFIG="${KAT_CONFIG:-kat_overrides_sota.yaml}"
# Serving limits. Defaults reproduce the 52.0% result (2026-08-22 full-pilot
# run) exactly: 1.22-1.41 GiB KV depending on Windows-side VRAM contention,
# ~90-98K tokens, both --workers 2 rollouts hold most or all of a full-length
# context. MAXLEN=32768 MAXSEQS=8 reproduces the original 40.0% baseline.
MAXLEN="${MAXLEN:-49152}"
MAXSEQS="${MAXSEQS:-2}"
STOCK=~/swebench-env/lib/python3.12/site-packages/minisweagent/config/benchmarks/swebench.yaml
LOG=~/kat_serve.log

cleanup () {
  if [ -n "${SERVER_PID:-}" ]; then
    echo "=== stopping server ($SERVER_PID) ==="
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
  fi
  # vLLM's engine renames its process to VLLM::EngineCore, so it does NOT match
  # "vllm serve" and survives killing the parent. A leaked engine holds the whole
  # weight allocation and silently starves the next run's KV cache - observed
  # 2026-08-21, where it made a 49152 probe fail with only 14.35 GiB free.
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null
  sleep 5
  local held
  held=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
  echo "=== GPU after teardown: ${held:-unknown} ==="
}
trap cleanup EXIT INT TERM

echo "=== starting vLLM (full log -> $LOG) ==="
echo "    max_model_len=$MAXLEN  max_num_seqs=$MAXSEQS  config=$KAT_CONFIG"
~/vllm-env/bin/vllm serve "$MODEL" \
  --served-model-name kat-16gb --port "$PORT" \
  --max-model-len "$MAXLEN" --max-num-seqs "$MAXSEQS" \
  --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --enable-prefix-caching --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_capture_sizes":[1,2],"cudagraph_mode":"PIECEWISE"}' \
  --language-model-only > "$LOG" 2>&1 &
SERVER_PID=$!

READY=0
for i in $(seq 1 150); do
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "!! server died:"; grep -aE "AssertionError|ValueError|RuntimeError" "$LOG" | head -3; exit 1; }
  curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200 && { READY=1; break; }
  sleep 4
done
[ "$READY" = 1 ] || { echo "!! timed out"; tail -20 "$LOG"; exit 1; }
echo "   ready"
grep -a "GPU KV cache size\|Maximum concurrency" "$LOG" | tail -2 | sed 's/^/   /'

echo
echo "=== PREFLIGHT: the agent's exact model path, through litellm ==="
"$ENVBIN/python" ~/preflight_litellm.py || { echo "!! aborting before spending containers or hours"; exit 1; }

echo
echo "=== PREFLIGHT: docker ==="
docker run --rm alpine:latest echo docker_ok 2>/dev/null | grep -q docker_ok \
  || { echo "!! docker run failed"; exit 1; }
echo "   ok"
df -h /var/lib/docker | tail -1 | sed 's/^/   /'

echo
echo "=== ROLLOUT: $N instances -> $OUT (config: $KAT_CONFIG) ==="
mkdir -p "$OUT"
cd "$CFGDIR" || exit 1
LITELLM_MODEL_REGISTRY_PATH="$CFGDIR/registry.json" \
"$ENVBIN/mini-extra" swebench \
  --subset verified --split test --shuffle --slice "0:$N" --workers 2 \
  -o "$OUT" -c "$STOCK" -c "$CFGDIR/$KAT_CONFIG" 2>&1 | tail -25

echo
echo "=== prefix cache hit counters during the rollout ==="
curl -s "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
  | grep -E "vllm:prefix_cache_(queries|hits)_total\{" | sed 's/^/   /'

echo
echo "=== ARTIFACTS (this stack returns 0 on failure routinely) ==="
if [ -f "$OUT/preds.json" ]; then
  "$ENVBIN/python" - "$OUT/preds.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("   instances with predictions:", len(d))
for k, v in d.items():
    p = (v.get("model_patch") or "").strip()
    print(f"     {k:<34} patch {len(p):>6} chars")
PY
else
  echo "   !! no preds.json"; ls -la "$OUT" | head
fi
