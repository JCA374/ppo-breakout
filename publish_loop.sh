#!/bin/bash
# Rebuild and push the site every 15 minutes (renders run with nice). Stop: kill the PID in publish_loop.pid.
# Also a watchdog: if the main run's supervisor died (e.g. after a reboot) and the run isn't DONE, restart it.
cd "$(dirname "$0")"
mkdir -p logs
echo $$ > publish_loop.pid
while true; do
  if [ ! -e runs/main/DONE ] && ! pgrep -f "supervise.sh main" > /dev/null; then
    echo "$(date '+%F %T') watchdog: restarting supervise.sh main" >> logs/publish.log
    setsid nohup ./supervise.sh main --torch-threads 4 > /dev/null 2>&1 < /dev/null &
  fi
  nice -n 15 .venv/bin/python publish.py --push >> logs/publish.log 2>&1
  sleep 900
done
