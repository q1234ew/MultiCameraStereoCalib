#!/bin/bash
# Restricted SSH command. No path/device supplied by clients; always restore owned service.
set -euo pipefail
case "${SSH_ORIGINAL_COMMAND:-}" in
  'photo 8080') port=8080;;
  'photo 8082') port=8082;;
  *) echo 'Unsupported photo command' >&2; exit 2;;
esac
exec 2>>/var/log/multicamera-photo.log
lock=/run/multicamera-photo.lock
mkdir "$lock" 2>/dev/null || { echo 'Photo already running' >&2; exit 3; }
work=$(mktemp -d /tmp/multicamera-photo.XXXXXX)
restore_needed=0
cleanup() {
  code=$?
  trap - EXIT HUP INT TERM
  if [ "$restore_needed" = 1 ]; then
    /etc/init.d/S90fusion-sensor start >&2 || code=4
    for saved in "$work"/video*.ctrl; do
      [ -f "$saved" ] || continue
      dev="/dev/$(basename "$saved" .ctrl)"
      while IFS= read -r pair; do
        timeout 3 v4l2-ctl -d "$dev" --set-ctrl "$pair" >&2 || code=4
      done < "$saved"
    done
  fi
  # Output only after recovery; callers reject a failed recovery even if a JPEG was captured.
  if [ "$code" = 0 ] && [ -s "$work/photo.jpg" ]; then cat "$work/photo.jpg"; fi
  rm -rf "$work"
  rmdir "$lock"
  exit "$code"
}
trap cleanup EXIT
trap 'exit 5' HUP INT TERM
/etc/init.d/S90fusion-sensor status >&2
# Read actual current stream bindings, including role swaps; never assume video node numbering.
ps -eo args | awk -v port="$port" '
 /ustreamer.bin/ && /--device / {dev="";p="";for(i=1;i<=NF;i++){if($i=="--device")dev=$(i+1);if($i=="--port")p=$(i+1)};if(p==port)print dev}' > "$work/target"
[ "$(wc -l < "$work/target")" = 1 ]
dev=$(cat "$work/target")
case "$dev" in /dev/video[0-9]*) ;; *) exit 2;; esac
ps -eo args | awk '/ustreamer.bin/ && /--device / {for(i=1;i<=NF;i++)if($i=="--device")print $(i+1)}' > "$work/devices"
while IFS= read -r node; do
  v4l2-ctl -d "$node" --list-ctrls | awk '
  /value=/ && !/inactive/ && !/read-only/ {name=$1;for(i=1;i<=NF;i++)if($i~/^value=/){sub(/^value=/,"",$i); print name"="$i}}' > "$work/$(basename "$node").ctrl"
done < "$work/devices"
ir=$(wget -qO- -T 3 http://127.0.0.1:8081/api/ir/config)
export IR_SCALE=$(printf '%s' "$ir" | sed -n 's/.*"scale":\([1-4]\).*/\1/p')
export IR_COLORMAP=$(printf '%s' "$ir" | sed -n 's/.*"colormap":"\([a-z]*\)".*/\1/p')
case "$IR_SCALE:$IR_COLORMAP" in [1-4]:inferno|[1-4]:turbo|[1-4]:jet|[1-4]:viridis) ;; *) exit 2;; esac
if [[ "$ir" == *'"overlay":true'* ]]; then export IR_OVERLAY=1; else export IR_OVERLAY=0; fi
# Unsupported extra IR rendering states must not silently change during recovery.
[[ "$ir" == *'"vmin":null'* && "$ir" == *'"vmax":null'* && "$ir" == *'"show_ir":false'* ]]
restore_needed=1
/etc/init.d/S90fusion-sensor stop >&2
timeout 15 v4l2-ctl -d "$dev" --set-fmt-video=width=1920,height=1080,pixelformat=MJPG \
  --set-parm=15 --stream-mmap=3 --stream-skip=4 --stream-count=1 --stream-to="$work/photo.jpg" >&2
