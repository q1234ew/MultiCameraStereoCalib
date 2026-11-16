#!/usr/bin/env bash
# Isolated trace instance only. Never changes global tracing or camera controls.
set -euo pipefail
instance="/sys/kernel/debug/tracing/instances/multicamera-clock-$$"
[[ -d /sys/kernel/debug/tracing/instances ]]
mkdir "$instance"
cleanup() {
    echo 0 > "$instance/tracing_on" 2>/dev/null || true
    echo 0 > "$instance/events/v4l2/v4l2_dqbuf/enable" 2>/dev/null || true
    rmdir "$instance" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 1' TERM INT
echo mono > "$instance/trace_clock"
echo 1 > "$instance/events/v4l2/v4l2_dqbuf/enable"
echo 1 > "$instance/tracing_on"
sleep 5
echo 0 > "$instance/tracing_on"
cat "$instance/trace"
