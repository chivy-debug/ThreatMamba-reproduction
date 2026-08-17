#!/usr/bin/env bash
# Stage 2 - start the IOCHunter daemon in the background (resumable).
# Usage:  bash scripts/run_enrichment_daemon.sh          # start in the background
#         bash scripts/run_enrichment_daemon.sh stop     # stop the daemon
#         bash scripts/run_enrichment_daemon.sh status   # show progress
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
PIDFILE=outputs/enrich_daemon.pid
mkdir -p outputs

case "${1:-start}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "daemon already running (PID $(cat $PIDFILE))"; exit 0
    fi
    nohup "$PY" -m src.ioc_hunter.runner >> outputs/enrich_daemon.out 2>&1 &
    echo $! > "$PIDFILE"
    echo "daemon started in the background, PID $(cat $PIDFILE). Log: outputs/enrich.log"
    ;;
  stop)
    [ -f "$PIDFILE" ] && kill "$(cat $PIDFILE)" 2>/dev/null \
      && echo "SIGTERM sent (the daemon saves its state and exits)" || echo "no daemon running"
    ;;
  status)
    "$PY" scripts/check_enrichment.py
    ;;
  *) echo "usage: $0 [start|stop|status]"; exit 1;;
esac
