#!/bin/sh
set -eu
APP="@APP@"
INIT=/etc/init.d/S92multicamera-calibration
WEB=/opt/fusion-sensor/web/index.html
cmp -s "$INIT" "$APP/service.init" || { echo "Init changed; inspect before rollback" >&2; exit 1; }
if [ -f "$APP/dashboard-original.html" ]; then
    cmp -s "$WEB" "$APP/dashboard.html" || { echo "Dashboard changed; will not overwrite" >&2; exit 1; }
    echo "@DASHBOARD_SHA@  $APP/dashboard-original.html" | sha256sum -c -
fi
"$INIT" stop
mv "$INIT" "$APP/disabled-init"
if [ -f "$APP/dashboard-original.html" ]; then
    [ ! -e /opt/fusion-sensor/web/.multicamera-index-rollback ]
    cp "$APP/dashboard-original.html" /opt/fusion-sensor/web/.multicamera-index-rollback
    mv /opt/fusion-sensor/web/.multicamera-index-rollback "$WEB"
fi
echo "New service disabled; original dashboard restored; payload and calibration retained"
