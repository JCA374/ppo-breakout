#!/bin/bash
# Narrate several videos one after another (waits for a running podcast publish first).
cd "$(dirname "$0")/.."
while pgrep -f "project-podcast/scripts/publish.sh" > /dev/null; do sleep 15; done
for n in "$@"; do video/narrate.sh $n > logs/narrate_$n.log 2>&1; done
