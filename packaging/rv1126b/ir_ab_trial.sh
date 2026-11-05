#!/bin/bash
# Authorized device-3 IR-only comparison. Always restore the original executable.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ORIGINAL=/opt/fusion-sensor/src/ustreamer.bin
CANDIDATE="$HERE/ustreamer-reference.bin"
EXPECTED_ORIGINAL=ec0c3fa58271f07110b9d17ada4f4195abde29ee04338a976de6ef9c909e03cf
EXPECTED_CANDIDATE=24b7063a69b77640865a6e3a6e611ba19d2eb9c840ce8da749e054eedf563803
SCALE=${1:?IR scale required}
COLORMAP=${2:?IR colormap required}
case "$SCALE" in 1|2|3|4) ;; *) exit 1;; esac
case "$COLORMAP" in inferno|turbo|jet|viridis) ;; *) exit 1;; esac
grep -q '^92:6f:fb:29:99:c4$' /sys/class/net/*/address
echo "$EXPECTED_ORIGINAL  $ORIGINAL" | sha256sum -c -
echo "$EXPECTED_CANDIDATE  $CANDIDATE" | sha256sum -c -
[ ! -e "$HERE/TRIAL_STARTED" ]
wget -qO "$HERE/config-at-stop.json" http://127.0.0.1:8081/api/ir/config
cmp -s "$HERE/baseline-config.json" "$HERE/config-at-stop.json"

rgb_identity() {
    for p in /proc/[0-9]*; do
        [ "$(readlink "$p/exe" 2>/dev/null)" = "$ORIGINAL" ] || continue
        cmd=$(tr '\000' ' ' <"$p/cmdline" 2>/dev/null)
        case "$cmd" in
            *"--port 8080 "*|*"--port 8082 "*) echo "$p $(awk '{print $22}' "$p/stat")" ;;
        esac
    done | sort
}
rgb_identity >"$HERE/rgb-before.txt"
[ "$(wc -l <"$HERE/rgb-before.txt")" -eq 2 ]
original_pid=
for p in /proc/[0-9]*; do
    [ "$(readlink "$p/exe" 2>/dev/null)" = "$ORIGINAL" ] || continue
    cmd=$(tr '\000' ' ' <"$p/cmdline" 2>/dev/null)
    case "$cmd" in
        *"--usb-device 04b4:f7f7 "*"--port 8081 "*)
            [ -z "$original_pid" ] || exit 1
            original_pid=${p##*/}
            cp "$p/cmdline" "$HERE/original-cmdline.bin"
            ;;
    esac
done
[ -n "$original_pid" ]
args=(--usb-device 04b4:f7f7 --ir-thermal --ir-colormap "$COLORMAP" --ir-scale "$SCALE"
      --host 0.0.0.0 --port 8081 --static /opt/fusion-sensor/web --ir-overlay)
candidate_pid=
baseline_stopped=0
cleanup() {
    trap - EXIT HUP INT TERM
    set +e
    if [ -n "$candidate_pid" ] && [ "$(readlink "/proc/$candidate_pid/exe" 2>/dev/null)" = "$CANDIDATE" ]; then
        kill -TERM "$candidate_pid"
        for unused in $(seq 1 50); do
            kill -0 "$candidate_pid" 2>/dev/null || break
            sleep 0.1
        done
        if kill -0 "$candidate_pid" 2>/dev/null; then kill -KILL "$candidate_pid"; fi
        wait "$candidate_pid" 2>/dev/null
    fi
    if [ "$baseline_stopped" = 1 ] && ! kill -0 "$original_pid" 2>/dev/null; then
        /usr/bin/setsid "$ORIGINAL" "${args[@]}" >>"$HERE/original-restored.log" 2>&1 </dev/null &
        restored=$!
        echo "$restored" > /var/run/fusion-sensor-ir-restored.pid
        sleep 3
        if ! kill -0 "$restored" 2>/dev/null; then
            echo RESTORE_FAILED
            return 1
        fi
        wget -qO "$HERE/config-restored.json" http://127.0.0.1:8081/api/ir/config
        if ! cmp -s "$HERE/baseline-config.json" "$HERE/config-restored.json"; then
            echo RESTORE_CONFIG_MISMATCH
            return 1
        fi
        echo ORIGINAL_IR_RESTORED
    fi
    rgb_identity >"$HERE/rgb-after.txt"
    if ! cmp -s "$HERE/rgb-before.txt" "$HERE/rgb-after.txt"; then
        echo RGB_IDENTITY_CHANGED
        return 1
    fi
    echo RGB_PROCESSES_UNCHANGED
    touch "$HERE/TRIAL_RESTORED"
}
trap cleanup EXIT
trap 'exit 130' HUP INT TERM
touch "$HERE/TRIAL_STARTED"
baseline_stopped=1
kill -TERM "$original_pid"
for unused in $(seq 1 50); do
    kill -0 "$original_pid" 2>/dev/null || break
    sleep 0.1
done
if kill -0 "$original_pid" 2>/dev/null; then echo ORIGINAL_DID_NOT_STOP; exit 1; fi
/usr/bin/setsid "$CANDIDATE" "${args[@]}" >>"$HERE/candidate.log" 2>&1 </dev/null &
candidate_pid=$!
echo "$candidate_pid" >"$HERE/candidate.pid"
sleep 3
kill -0 "$candidate_pid"
echo CANDIDATE_READY
touch "$HERE/CANDIDATE_READY"
for n in 1 2 3 4 5 6; do
    cat /proc/uptime >"$HERE/candidate-$n-uptime.txt"
    wget -T 3 -qO "$HERE/candidate-$n.jpg" http://127.0.0.1:8081/snapshot
    wget -T 3 -qO "$HERE/thermal-$n.json" http://127.0.0.1:8081/api/ir/thermal
    sleep 0.2
done
sleep 5
