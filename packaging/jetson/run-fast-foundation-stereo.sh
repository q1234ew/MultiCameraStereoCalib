#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_root="$project_root/models/fast-foundation-stereo_476f4249561f"
exec "$project_root/.fast-stereo-venv/bin/python" \
  "$project_root/scripts/jetson/fast_foundation_smoke.py" \
  --root "$model_root" \
  --checkpoint "$model_root/weights/20-30-48/model_best_bp2_serialize.pth" \
  "$@"
