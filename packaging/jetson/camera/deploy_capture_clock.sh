#!/usr/bin/env bash
# Run on a camera after uploading the proven artifact to BACKUP. RGB-only restart.
set -euo pipefail
BASE_SHA=${CAPTURE_CLOCK_EXPECTED_SHA:-24b7063a69b77640865a6e3a6e611ba19d2eb9c840ce8da749e054eedf563803}
NEW_SHA=${CAPTURE_CLOCK_NEW_SHA:?Provide the manifest binary_sha256}
APP=/opt/fusion-sensor
BACKUP=${CAPTURE_CLOCK_BACKUP:?Provide a new absolute backup directory}
[[ $BACKUP == "$APP/backups/"* && $BACKUP != *..* ]]
[[ $BASE_SHA =~ ^[a-f0-9]{64}$ && $NEW_SHA =~ ^[a-f0-9]{64}$ ]]
BIN="$APP/src/ustreamer.bin"
exec 9>/run/multicamera-capture-clock.lock
flock -n 9
[[ $(sha256sum "$BIN" | awk '{print $1}') == "$BASE_SHA" ]]
[[ $(sha256sum "$BACKUP/ustreamer.timestamp.bin" | awk '{print $1}') == "$NEW_SHA" ]]
[[ ! -e "$BACKUP/ustreamer.original.bin" ]] # Refuse to overwrite a previous backup.
LD_LIBRARY_PATH="$APP/lib" "$BACKUP/native-test"
cp -p "$BIN" "$BACKUP/ustreamer.original.bin"
config_files=("$APP/fusion-supervisor.sh" "$APP/start_fusion.sh")
if [[ -f /etc/fusion-sensor.conf ]]; then
    config_files+=(/etc/fusion-sensor.conf)
else
    : > "$BACKUP/config-absent-before"
fi
sha256sum "${config_files[@]}" > "$BACKUP/config-before.sha256"
: > "$BACKUP/infrared-before.tsv"
: > "$BACKUP/rgb-before.tsv"
for proc in /proc/[0-9]*; do
    [[ -r $proc/cmdline ]] || continue
    mapfile -d '' -t args < "$proc/cmdline"
    (( ${#args[@]} > 2 )) || continue
    case ${args[0]} in */ustreamer.bin) ;; *) continue ;; esac
    pid=${proc##*/}
    case ${args[1]} in
        --device)
            [[ ${args[0]} == "$BIN" ]] || continue
            ppid=$(awk '{print $4}' "$proc/stat")
            token=$(awk '{print $22}' "$proc/stat")
            cat "$proc/cmdline" > "$BACKUP/rgb-$pid.cmdline"
            printf '%s %s %s\n' "$pid" "$ppid" "$token" >> "$BACKUP/rgb-before.tsv" ;;
        --usb-device|--uvc-device)
            printf '%s %s %s\n' "$pid" "$(awk '{print $22}' "$proc/stat")" \
                "$(sha256sum "$proc/exe" | awk '{print $1}')" >> "$BACKUP/infrared-before.tsv" ;;
    esac
done
[[ $(wc -l < "$BACKUP/rgb-before.tsv") == 2 ]]
[[ $(wc -l < "$BACKUP/infrared-before.tsv") == 1 ]]
# Check every parent before changing anything. Per-channel supervisors restart
# their RGB child only. Unsupervised RGB is relaunched with its exact argv.
while read -r pid ppid token; do
    if [[ $ppid != 1 ]]; then
        tr '\000' '\n' < "/proc/$ppid/cmdline" | grep -Fxq "$APP/fusion-supervisor.sh"
    fi
done < "$BACKUP/rgb-before.tsv"
cp -p "$BACKUP/ustreamer.timestamp.bin" "$BIN.capture-new"
mv "$BIN.capture-new" "$BIN"
while read -r pid ppid token; do
    [[ $(awk '{print $22}' "/proc/$pid/stat") == "$token" ]]
    kill -TERM "$pid"
    for ((i=0; i<40; ++i)); do
        [[ -r /proc/$pid/stat ]] || break
        [[ $(awk '{print $3}' "/proc/$pid/stat") == Z ]] && break
        sleep .1
    done
    if [[ -r /proc/$pid/stat && $(awk '{print $3}' "/proc/$pid/stat") != Z ]]; then
        echo "RGB process did not stop: $pid; restore original binary from $BACKUP" >&2
        exit 1
    fi
    if [[ $ppid == 1 ]]; then
        mapfile -d '' -t args < "$BACKUP/rgb-$pid.cmdline"
        LD_LIBRARY_PATH="$APP/lib" nohup "${args[@]}" 9>&- \
            > "$BACKUP/rgb-$pid-restarted.log" 2>&1 < /dev/null &
        echo "$!" >> "$BACKUP/rgb-restarted.pids"
    fi
    sleep 1
done < "$BACKUP/rgb-before.tsv"
while read -r pid token hash; do
    [[ $(awk '{print $22}' "/proc/$pid/stat") == "$token" ]]
    [[ $(sha256sum "/proc/$pid/exe" | awk '{print $1}') == "$hash" ]]
done < "$BACKUP/infrared-before.tsv"
sha256sum -c "$BACKUP/config-before.sha256"
if [[ -f $BACKUP/config-absent-before ]]; then [[ ! -e /etc/fusion-sensor.conf ]]; fi
printf '%s\n' "installed=$NEW_SHA" "infrared_process_unchanged=true" \
    "rgb_restart_only=true" > "$BACKUP/deploy-result.txt"
cat "$BACKUP/deploy-result.txt"
