#!/usr/bin/env bash
# Print the exact version of every component this pipeline depends on.
#
# This project lost a release once to unpinned dependencies: four environments,
# each load-bearing, none recorded in the repo. Run this after any environment
# change and paste the output into docs/environment.md so the table stays true.
#
# Prints, never mutates.
set -uo pipefail

echo "captured: $(date -Iseconds)"
echo

echo "=== hardware ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null \
  || echo "  nvidia-smi unavailable"
echo

echo "=== reap (the fork is load-bearing: renormalization fix) ==="
if [ -d "$HOME/reap-cuda/.git" ]; then
  echo "  remote : $(git -C "$HOME/reap-cuda" remote get-url origin)"
  echo "  branch : $(git -C "$HOME/reap-cuda" rev-parse --abbrev-ref HEAD)"
  echo "  commit : $(git -C "$HOME/reap-cuda" rev-parse HEAD)"
  dirty=$(git -C "$HOME/reap-cuda" status --porcelain | wc -l)
  [ "$dirty" -gt 0 ] && echo "  !! $dirty uncommitted change(s) - the build is not reproducible from this commit alone"
else
  echo "  !! ~/reap-cuda missing"
fi
echo

echo "=== vLLM source build ==="
if [ -d "$HOME/vllm-src/.git" ]; then
  echo "  tag/commit : $(git -C "$HOME/vllm-src" describe --tags --always 2>/dev/null)"
else
  echo "  ~/vllm-src not a git checkout"
fi
echo

PKGS='^(torch|transformers|vllm|llmcompressor|compressed-tensors|lm.eval|datasets|accelerate|safetensors|reap|flashinfer.*|evaluate) '
for env in reap-cuda-env quant-env vllm-env swebench-env; do
  [ -d "$HOME/$env" ] || continue
  echo "=== ~/$env ==="
  "$HOME/$env/bin/pip" list 2>/dev/null | grep -iE "$PKGS" | sed 's/^/  /' || echo "  (none of interest)"
  echo
done

echo "=== released checkpoint ==="
RC="$HOME/models/kat-50pct-nvfp4a16-renorm-stripped"
if [ -f "$RC/model.safetensors" ]; then
  sz=$(stat -c%s "$RC/model.safetensors")
  printf '  model.safetensors: %s bytes (%.4f GiB)\n' "$sz" "$(echo "$sz/1073741824" | bc -l)"
else
  echo "  not present locally"
fi
