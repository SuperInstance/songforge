#!/usr/bin/env bash
# Session 71 variant: force CPU (num_gpu 0) — GPU held by midi_studio.py
# Usage: session71_lyrics_cpu.sh <model> <temperature> <outfile> [prompt-file or inline]
set -euo pipefail
MODEL="$1"; TEMP="$2"; OUTFILE="$3"; PROMPT="${4:-}"

if [ -z "$PROMPT" ]; then PROMPT="$(cat /dev/stdin)"; fi
if [ -f "$PROMPT" ]; then PROMPT="$(cat "$PROMPT")"; fi

ESCAPED="$(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' | sed 's/^"//;s/"$//')"

PAYLOAD="{\"model\":\"$MODEL\",\"prompt\":\"$ESCAPED\",\"stream\":false,\"options\":{\"temperature\":$TEMP,\"num_gpu\":0}}"

curl -s http://127.0.0.1:11434/api/generate -d "$PAYLOAD" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])' > "$OUTFILE"
echo "wrote $OUTFILE ($(wc -w < "$OUTFILE") words)"
