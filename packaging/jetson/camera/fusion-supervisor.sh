#!/bin/bash
# SysV camera supervisor. No changes to camera ordering, calibration or capture settings.
set -u
APP_DIR=${FUSION_APP_DIR:-/opt/fusion-sensor}
RUN_DIR=${FUSION_RUN_DIR:-/var/run}
PIDFILE="$RUN_DIR/fusion-sensor.pid"
STATEFILE="$RUN_DIR/fusion-sensor.state"
INTERFACE=${FUSION_NETWORK_INTERFACE:-eth0}
INTERVAL=${FUSION_CHECK_INTERVAL:-2}
NETWORK_INTERVAL=${FUSION_NETWORK_RETRY_INTERVAL:-5}
GRACE=${FUSION_STARTUP_GRACE:-25}
MAX_FAILURES=${FUSION_MAX_FAILURES:-3}
PORTS=${FUSION_HEALTH_PORTS:-"8080 8081 8082"}
child=""
cd "$APP_DIR" || exit 1
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/fusion-sensor.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" >"$PIDFILE.tmp"
mv "$PIDFILE.tmp" "$PIDFILE"

state() {
    printf '%s\n' "$1" >"$STATEFILE.tmp"
    mv "$STATEFILE.tmp" "$STATEFILE"
}
log() { printf '[SUPERVISOR] %s %s\n' "$(date -u +%FT%TZ)" "$*"; }
network_ready() {
    ip -o link show dev "$INTERFACE" 2>/dev/null | grep -q 'LOWER_UP' || return 1
    ip -4 -o addr show dev "$INTERFACE" scope global 2>/dev/null |
        awk '$3 == "inet" && $4 !~ /^169[.]254[.]/ && $4 !~ /^127[.]/ {found=1}
             END {exit !found}'
}
streams_ready() {
    local port
    for port in $PORTS; do
        # Verify HTTP responds, including partial child failures hidden by launcher's wait-all.
        timeout 3 wget -q -T 2 -O /dev/null "http://127.0.0.1:$port/snapshot" || return 1
    done
}
stop_child() {
    local attempt
    if [ -n "$child" ]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        for attempt in {1..20}; do
            kill -0 -- "-$child" 2>/dev/null || break
            sleep .1
        done
        kill -KILL -- "-$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
        child=""
    fi
}
cleanup() {
    trap - EXIT TERM INT
    state stopping
    stop_child
    if [ "$(cat "$PIDFILE" 2>/dev/null)" = "$$" ]; then
        rm -f "$PIDFILE" "$STATEFILE"
    fi
    log 'supervisor stopped; automatic restart disabled until start'
}
trap cleanup EXIT
trap 'exit 0' TERM INT
backoff=2
while true; do
    state waiting_network
    while ! network_ready; do
        log "waiting for carrier and global IPv4 on $INTERFACE"
        sleep "$NETWORK_INTERVAL"
    done
    state starting
    log 'network ready; starting camera process group'
    # The supervisor lock must not be retained by capture children after supervisor exit.
    setsid bash "$APP_DIR/start_fusion.sh" 9>&- &
    child=$!
    started=$SECONDS
    failures=0
    healthy_since=0
    while kill -0 "$child" 2>/dev/null; do
        if streams_ready; then
            if [ "$healthy_since" -eq 0 ]; then
                healthy_since=$SECONDS
                state healthy
                log "all capture endpoints ready (launcher $child)"
            fi
            failures=0
            if [ "$((SECONDS - healthy_since))" -ge 60 ]; then backoff=2; fi
        elif [ "$((SECONDS - started))" -ge "$GRACE" ]; then
            healthy_since=0
            state degraded
            failures=$((failures + 1))
            log "capture health failed ($failures/$MAX_FAILURES)"
            if [ "$failures" -ge "$MAX_FAILURES" ]; then break; fi
        fi
        sleep "$INTERVAL"
    done
    log "capture exited or health failed; restarting in ${backoff}s"
    stop_child
    state backoff
    sleep "$backoff"
    backoff=$((backoff * 2))
    if [ "$backoff" -gt 30 ]; then backoff=30; fi
done
