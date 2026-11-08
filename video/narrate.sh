#!/bin/bash
# Narrate a video script with the voice-text pipeline: new -> scan -> prep -> narrate -> listen (Whisper) -> export -> status.
#   video/narrate.sh <name>      (video/<name>/script.md -> voice/<name>/audio/voiceover/*.mp3 + timings.json)
cd "$(dirname "$0")/.."
V=~/.claude/skills/voice-text/scripts/voice.sh
N=$1
if [ -d voice/$N ]; then $V new video/$N/script.md --kind voiceover --slug $N --force; else $V new video/$N/script.md --kind voiceover --slug $N; fi
cp voice/glossary.md voice/$N/glossary.md
$V scan voice/$N
$V prep voice/$N
nice -n 10 $V narrate voice/$N
nice -n 10 $V listen voice/$N
$V export voice/$N
$V status voice/$N
