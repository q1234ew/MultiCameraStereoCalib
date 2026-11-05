#!/bin/sh
# Candidate files only: never alter services, current links or camera controls.
set -eu
HERE=$(CDPATH= cd "$(dirname "$0")" && pwd)
DEVICE='@DEVICE_ID@'
VERSION='@VERSION@'
EXPECTED_MAC='@EXPECTED_MAC@'
ROOT=${1:-/opt/fusion-sensor/calibration-candidates}
case "$ROOT" in /*) ;; *) echo 'Destination must be absolute' >&2; exit 1;; esac
if [ "$#" -gt 1 ]; then echo 'Usage: sh install.sh [candidate-root]' >&2; exit 1; fi
command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 1; }
MATCH=0
for entry in /sys/class/net/*/address; do
    [ -f "$entry" ] || continue
    MAC=$(tr 'A-F' 'a-f' < "$entry")
    if [ "$MAC" = "$EXPECTED_MAC" ]; then MATCH=1; fi
done
[ "$MATCH" = 1 ] || { echo 'Device MAC mismatch; no files installed' >&2; exit 1; }
[ -z "$(find "$HERE/payload" -type l -print)" ] || {
    echo 'Payload symlinks forbidden' >&2; exit 1;
}
(cd "$HERE/payload" && sha256sum -c "$HERE/SHA256SUMS.txt")
CHECK="$ROOT/$DEVICE"
while [ "$CHECK" != / ]; do
    [ ! -L "$CHECK" ] || { echo 'Symlink destination forbidden' >&2; exit 1; }
    CHECK=$(dirname "$CHECK")
done
TARGET="$ROOT/$DEVICE/$VERSION"
mkdir -p "$ROOT/$DEVICE"
mkdir "$TARGET" || { echo 'Refusing an existing candidate version' >&2; exit 1; }
# A failed copy remains visibly incomplete; never remove or replace an existing version.
cp "$HERE/payload/"* "$TARGET/"
cp "$HERE/SHA256SUMS.txt" "$TARGET/"
if [ -f "$TARGET/compatibility_test" ]; then chmod 755 "$TARGET/compatibility_test"; fi
(cd "$TARGET" && sha256sum -c SHA256SUMS.txt)
printf '%s\n' 'candidate_only_not_activated' > "$TARGET/candidate-status.txt"
printf 'Installed candidate only: %s\nNo parameters activated; no services restarted.\n' "$TARGET"
