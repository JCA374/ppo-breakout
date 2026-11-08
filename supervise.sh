#!/bin/bash
# Keep a training run alive: restart from checkpoints/<run>/latest.pt after any crash, until runs/<run>/DONE exists.
#   setsid nohup ./supervise.sh main [extra train.py args] > /dev/null 2>&1 < /dev/null &
cd "$(dirname "$0")"
RUN=$1; shift
mkdir -p logs runs/$RUN
echo $$ > runs/$RUN/supervisor.pid
while [ ! -e runs/$RUN/DONE ]; do
  .venv/bin/python ppo/train.py --run $RUN "$@" >> logs/train_$RUN.log 2>&1
  code=$?
  if [ ! -e runs/$RUN/DONE ]; then
    echo "{\"t\": \"$(date "+%F %T")\", \"kind\": \"crash\", \"note\": \"train.py exited with code $code; restarting from the last checkpoint\"}" >> runs/$RUN/events.jsonl
    sleep 20
  fi
done
