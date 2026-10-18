#!/usr/bin/env bash
# Run on a Docker host after building the actual RunPod image; no GPU/token needed.
set -euo pipefail
IMAGE="${1:-peft-muon:ci}"
LOG_DIR="${2:-docker-checks}"
mkdir -p "$LOG_DIR"
docker image inspect "$IMAGE" > "$LOG_DIR/image-inspect.json"
python3 - "$LOG_DIR/image-inspect.json" <<'PY'
import json
import sys
image = json.load(open(sys.argv[1]))[0]
assert image['Architecture'] == 'amd64' and image['Os'] == 'linux'
assert image['Config']['Entrypoint'] == ['/usr/bin/tini', '--']
assert image['Config']['Cmd'] == ['bash', '/opt/muon/runpod/container-start.sh']
assert image['Config']['WorkingDir'] == '/opt/muon'
print('Image ID:', image['Id'])
PY

# Empty /workspace simulates a Pod volume. No host source/cache/token is mounted.
# Retain the real ENTRYPOINT; only CLI checks override CMD. No --gpus flag.
COMMON=(--rm --network none --workdir /tmp --tmpfs /workspace:rw
  -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=
  -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1
  -e HF_TOKEN= -e HUGGING_FACE_HUB_TOKEN=
  -e "EXPECTED_SOURCE_COMMIT=$(git rev-parse HEAD)")
docker run "${COMMON[@]}" "$IMAGE" /opt/venv/bin/python \
  /opt/muon/runpod/check_environment.py 2>&1 | tee "$LOG_DIR/environment.log"
docker run "${COMMON[@]}" "$IMAGE" /opt/venv/bin/python \
  /opt/muon/runpod/validate.py --help | tee "$LOG_DIR/validate-help.txt"
docker run "${COMMON[@]}" "$IMAGE" bash /opt/muon/runpod/run.sh --help \
  | tee "$LOG_DIR/runner-help.txt"
grep -q -- '--check-config' "$LOG_DIR/validate-help.txt"
grep -q -- '--check-config' "$LOG_DIR/runner-help.txt"
docker run "${COMMON[@]}" "$IMAGE" bash /opt/muon/runpod/run.sh --check-config \
  | tee "$LOG_DIR/default-config.json"
for profile in smoke performance; do
  docker run "${COMMON[@]}" "$IMAGE" bash /opt/muon/runpod/run.sh \
    --config "/opt/muon/runpod/configs/$profile.json" --check-config \
    | tee "$LOG_DIR/$profile-config.json"
done
cmp "$LOG_DIR/default-config.json" "$LOG_DIR/smoke-config.json"

# Invoke the unmodified default ENTRYPOINT + CMD, including shell/profile paths.
# Expected failure must be the explicit CUDA preflight, not import/path errors.
for profile in smoke performance all; do
  status=0
  timeout 120s docker run "${COMMON[@]}" -e "RUN_SUITE=$profile" "$IMAGE" \
    > "$LOG_DIR/startup-$profile.log" 2>&1 || status=$?
  cat "$LOG_DIR/startup-$profile.log"
  test "$status" -eq 1
  grep -Fq 'Expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES=0)' "$LOG_DIR/startup-$profile.log"
done
status=0
timeout 120s docker run "${COMMON[@]}" -e RUN_SUITE=invalid "$IMAGE" \
  > "$LOG_DIR/startup-invalid.log" 2>&1 || status=$?
test "$status" -eq 2
grep -Fq 'RUN_SUITE must be smoke, performance, or all' "$LOG_DIR/startup-invalid.log"
printf 'PASS: Docker environment, CLI, configs and missing-GPU guard; GPU training NOT tested.\n'
