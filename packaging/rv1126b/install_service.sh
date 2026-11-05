#!/bin/sh
set -eu
APP="@APP@"
CALIBRATION="@CALIBRATION@"
INIT=/etc/init.d/S92multicamera-calibration
HERE=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
cd "$HERE"
sha256sum -c SHA256SUMS.txt
[ -f "$CALIBRATION" ] || { echo "Missing installed calibration" >&2; exit 1; }
echo "@CALIBRATION_SHA@  $CALIBRATION" | sha256sum -c -
found=false
for address in /sys/class/net/*/address; do
    [ "$(cat "$address")" != "@MAC@" ] || found=true
done
[ "$found" = true ] || { echo "Device MAC mismatch" >&2; exit 1; }
[ ! -e "$APP" ] && [ ! -L "$APP" ] || { echo "Service version exists" >&2; exit 1; }
[ ! -e "$INIT" ] && [ ! -L "$INIT" ] || { echo "Init entry exists; will not overwrite" >&2; exit 1; }
WEB=/opt/fusion-sensor/web/index.html
if [ -f dashboard.html ]; then
    [ ! -L "$WEB" ] && [ ! -L /opt/fusion-sensor/web ]
    echo "@DASHBOARD_SHA@  $WEB" | sha256sum -c -
fi
parent=$(dirname "$APP")
p="$parent"
while [ "$p" != / ]; do
    [ ! -L "$p" ] || { echo "Symlink destination rejected" >&2; exit 1; }
    p=$(dirname "$p")
done
mkdir -p "$parent"
mkdir "$APP"
for name in calibration_service service.yml service.init install.sh rollback.sh panel.html THIRD_PARTY_NOTICES.txt SHA256SUMS.txt; do
    [ -f "$name" ] && [ ! -L "$name" ]
    cp "$name" "$APP/$name"
done
if [ -f dashboard.html ]; then
    cp dashboard.html dashboard-original.html "$APP/"
fi
chmod 755 "$APP/calibration_service" "$APP/service.init"
(cd "$APP" && sha256sum -c SHA256SUMS.txt)
# No existing init/config is replaced. Candidate files remain immutable.
cp "$APP/service.init" "$INIT"
chmod 755 "$INIT"
if ! "$INIT" start; then
    mv "$INIT" "$APP/failed-init"
    echo "Activation failed; new boot entry disabled, payload retained" >&2
    exit 1
fi
if [ -f dashboard.html ]; then
    # Preserve exact previous bytes and refuse concurrent edits before replacement.
    echo "@DASHBOARD_SHA@  $WEB" | sha256sum -c -
    [ ! -e /opt/fusion-sensor/web/.multicamera-index-new ]
    cp "$APP/dashboard.html" /opt/fusion-sensor/web/.multicamera-index-new
    mv /opt/fusion-sensor/web/.multicamera-index-new "$WEB"
fi
echo "Enabled on port 8090; source availability and physical accuracy are separate checks"
