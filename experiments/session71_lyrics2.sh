#!/usr/bin/env bash
# Session 71 lyric gen v2: CPU, capped output, explicit lyric format.
# Usage: session71_lyrics2.sh <model> <temperature> <outfile> <promptfile>
set -euo pipefail
MODEL="$1"; TEMP="$2"; OUTFILE="$3"; PROMPT_FILE="$4"

DESIGN="$(python3 -c "import sys,json; d=json.load(open('$PROMPT_FILE')); print(d['prompt'])")"
PROMPT="Write song lyrics for this song concept. Output ONLY the lyrics with [Verse], [Chorus], [Bridge] structure tags. No commentary, no title, no intro. 2 verses, 2 choruses, 1 bridge, one [Outro] line.
CONCEPT: $DESIGN"

ESCAPED="$(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' | sed 's/^"//;s/"$//')"

PAYLOAD="{\"model\":\"$MODEL\",\"prompt\":\"$ESCAPED\",\"stream\":false,\"options\":{\"temperature\":$TEMP,\"num_gpu\":0,\"num_predict\":1200,\"num_ctx\":8192}}"

curl -s --max-time 600 http://127.0.0.1:11434/api/generate -d "$PAYLOAD" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])' > "$OUTFILE"
echo "wrote $OUTFILE ($(wc -w < "$OUTFILE") words)"
