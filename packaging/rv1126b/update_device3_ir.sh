#!/bin/bash
# Pinned, device-3-only rollout; does not restart existing RGB processes.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=/opt/fusion-sensor/src/ustreamer.bin
REFERENCE=/opt/fusion-sensor/diagnostics/ir-sync-ab-20261105-2/ustreamer-reference.bin
OLD=ec0c3fa58271f07110b9d17ada4f4195abde29ee04338a976de6ef9c909e03cf
NEW=24b7063a69b77640865a6e3a6e611ba19d2eb9c840ce8da749e054eedf563803
STAGED=/opt/fusion-sensor/src/.ustreamer-ir-update-20261105-1
CALIBRATION=/opt/fusion-sensor/calibration-candidates/device_3/0.6.1-compat.20261105.2/calibration_mm.yml
SCALE=${1:?IR scale required}
COLORMAP=${2:?IR colormap required}
case "$SCALE" in 1|2|3|4) ;; *) exit 1;; esac
case "$COLORMAP" in inferno|turbo|jet|viridis) ;; *) exit 1;; esac
export LD_LIBRARY_PATH="/opt/fusion-sensor/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
grep -q '^92:6f:fb:29:99:c4$' /sys/class/net/*/address
echo "$OLD  $TARGET" | sha256sum -c -
echo "$NEW  $REFERENCE" | sha256sum -c -
"$REFERENCE" --version
[ ! -e "$HERE/UPDATE_STARTED" ] && [ ! -e "$STAGED" ]
[ ! -L "$TARGET" ] && [ ! -L "$STAGED" ]
wget -T 3 -qO "$HERE/config-at-stop.json" http://127.0.0.1:8081/api/ir/config
cmp -s "$HERE/baseline-config.json" "$HERE/config-at-stop.json"
sha256sum "$CALIBRATION" >"$HERE/calibration-before.sha256"
cp -p "$TARGET" "$HERE/ustreamer-original.bin"
echo "$OLD  $HERE/ustreamer-original.bin" | sha256sum -c -

original_pid=
: >"$HERE/rgb-before.txt"
for p in /proc/[0-9]*; do
    [ "$(readlink "$p/exe" 2>/dev/null)" = "$TARGET" ] || continue
    cmd=$(tr '\000' ' ' <"$p/cmdline" 2>/dev/null)
    case "$cmd" in
        *"--port 8080 "*|*"--port 8082 "*)
            echo "${p##*/} $(awk '{print $22}' "$p/stat")" >>"$HERE/rgb-before.txt"
            ;;
        *"--usb-device 04b4:f7f7 "*"--port 8081 "*)
            [ -z "$original_pid" ] || exit 1
            original_pid=${p##*/}
            cp "$p/cmdline" "$HERE/ir-original-cmdline.bin"
            ;;
    esac
done
[ "$(wc -l <"$HERE/rgb-before.txt")" -eq 2 ] && [ -n "$original_pid" ]
args=(--usb-device 04b4:f7f7 --ir-thermal --ir-colormap "$COLORMAP" --ir-scale "$SCALE"
      --host 0.0.0.0 --port 8081 --static /opt/fusion-sensor/web --ir-overlay)
committed=0
stopped=0
new_pid=
rollback() {
    trap - EXIT HUP INT TERM
    set +e
    [ "$committed" = 0 ] || return 0
    echo UPDATE_FAILED_ROLLING_BACK
    if [ -n "$new_pid" ] && kill -0 "$new_pid" 2>/dev/null; then
        kill -TERM "$new_pid"
        for unused in $(seq 1 50); do kill -0 "$new_pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$new_pid" 2>/dev/null; then kill -KILL "$new_pid"; fi
        wait "$new_pid" 2>/dev/null
    fi
    current=$(sha256sum "$TARGET" | awk '{print $1}')
    if [ "$current" = "$NEW" ]; then
        cp -p "$HERE/ustreamer-original.bin" "$STAGED"
        mv "$STAGED" "$TARGET"
    elif [ "$current" != "$OLD" ]; then
        echo TARGET_CHANGED_EXTERNALLY_MANUAL_RECOVERY_REQUIRED
        return 1
    fi
    if [ "$stopped" = 1 ] && ! kill -0 "$original_pid" 2>/dev/null; then
        /usr/bin/setsid "$TARGET" "${args[@]}" >>"$HERE/rollback-ir.log" 2>&1 </dev/null &
        echo "$!" >/var/run/fusion-sensor-ir-restored.pid
        sleep 3
        wget -T 3 -qO "$HERE/rollback-config.json" http://127.0.0.1:8081/api/ir/config
        cmp -s "$HERE/baseline-config.json" "$HERE/rollback-config.json" || return 1
    fi
    echo ORIGINAL_PROGRAM_RESTORED
}
trap rollback EXIT
trap 'exit 130' HUP INT TERM
touch "$HERE/UPDATE_STARTED"
cp "$REFERENCE" "$STAGED"
chmod 755 "$STAGED"
echo "$NEW  $STAGED" | sha256sum -c -
stopped=1
kill -TERM "$original_pid"
for unused in $(seq 1 50); do kill -0 "$original_pid" 2>/dev/null || break; sleep 0.1; done
if kill -0 "$original_pid" 2>/dev/null; then echo ORIGINAL_DID_NOT_STOP; exit 1; fi
mv "$STAGED" "$TARGET"
/usr/bin/setsid "$TARGET" "${args[@]}" >>"$HERE/new-ir.log" 2>&1 </dev/null &
new_pid=$!
sleep 3
kill -0 "$new_pid"
for n in 1 2 3; do
    wget -T 3 -qO "$HERE/thermal-$n.json" http://127.0.0.1:8081/api/ir/thermal
    grep -q '"ok":true' "$HERE/thermal-$n.json"
    wget -T 3 -qO "$HERE/snapshot-$n.jpg" http://127.0.0.1:8081/snapshot
    sleep 0.2
done
wget -T 3 -qO "$HERE/config-after.json" http://127.0.0.1:8081/api/ir/config
cmp -s "$HERE/baseline-config.json" "$HERE/config-after.json"
: >"$HERE/rgb-after.txt"
while read -r pid started; do
    kill -0 "$pid"
    echo "$pid $(awk '{print $22}' "/proc/$pid/stat")" >>"$HERE/rgb-after.txt"
    echo "$OLD  /proc/$pid/exe" | sha256sum -c -
done <"$HERE/rgb-before.txt"
cmp -s "$HERE/rgb-before.txt" "$HERE/rgb-after.txt"
sha256sum -c "$HERE/calibration-before.sha256"
echo "$NEW  $TARGET" | sha256sum -c -
echo "$NEW  /proc/$new_pid/exe" | sha256sum -c -
echo "$new_pid" >/var/run/fusion-sensor-ir.pid
if [ -f /var/run/fusion-sensor-ir-restored.pid ]; then
    mv /var/run/fusion-sensor-ir-restored.pid "$HERE/previous-restored-ir.pid"
fi
touch "$HERE/UPDATE_COMMITTED"
committed=1
echo PROGRAM_UPDATED_IR_RUNNING_RGB_PROCESSES_UNCHANGED
