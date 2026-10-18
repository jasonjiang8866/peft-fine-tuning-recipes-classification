#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SUITE="${RUN_SUITE:-smoke}"
case "$SUITE" in
  smoke|performance|all) ;;
  *) printf 'RUN_SUITE must be smoke, performance, or all\n' >&2; exit 2 ;;
esac
# mktemp prevents mixed/stale reports, including across container restarts.
mkdir -p "${RESULTS_ROOT:-/workspace/results}"
RUN_DIR="$(mktemp -d "${RESULTS_ROOT:-/workspace/results}/muon-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
printf 'Results: %s\n' "$RUN_DIR"
if [[ "$SUITE" == smoke || "$SUITE" == all ]]; then
  bash "$ROOT/runpod/run.sh" --config "$ROOT/runpod/configs/smoke.json" --output "$RUN_DIR/smoke"
fi
if [[ "$SUITE" == performance || "$SUITE" == all ]]; then
  bash "$ROOT/runpod/run.sh" --config "$ROOT/runpod/configs/performance.json" --output "$RUN_DIR/performance"
fi
printf 'Finished. Results: %s\nStop the Pod in RunPod after collecting results; process exit does not guarantee billing stops.\n' "$RUN_DIR"
