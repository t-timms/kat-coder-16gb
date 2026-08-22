#!/usr/bin/env bash
# Sync agent configs from the repo into the directory the runner actually reads.
set -uo pipefail
REPO="$HOME/kat-coder-16gb/scripts/swebench"
CFGDIR="$HOME/kat_swebench"
mkdir -p "$CFGDIR"
for f in kat_overrides.yaml kat_overrides_context_managed.yaml kat_overrides_sota.yaml kat_overrides_sota_presence_penalty.yaml registry.json; do
  if [ -f "$REPO/$f" ]; then
    cp "$REPO/$f" "$CFGDIR/$f"
    echo "  synced $f ($(stat -c%s "$CFGDIR/$f") bytes)"
  else
    echo "  !! missing in repo: $f"
  fi
done
echo
echo "=== CFGDIR now contains ==="
ls -la "$CFGDIR" | sed 's/^/  /'
