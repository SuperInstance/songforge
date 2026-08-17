#!/usr/bin/env bash
# Session 71 lyric gen v3: tiny model, truncated prompt, capped output — fast on CPU.
set -euo pipefail
MODEL="$1"; TEMP="$2"; OUTFILE="$3"; PROMPT_FILE="$4"

DESIGN="$(python3 -c "import sys,json; d=json.load(open('$PROMPT_FILE')); print(d['prompt'][:400])")"
PROMPT="Write lyrics with [Verse] [Chorus] [Bridge] tags for: $DESIGN"

ESCAPED="$(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' | sed 's/^"//;s/"$//')"
PAYLOAD="{\"model\":\"$MODEL\",\"prompt\":\"$ESCAPED\",\"stream\":false,\"options\":{\"temperature\":$TEMP,\"num_gpu\":0,\"num_predict\":800,\"num_ctx\":4096}}"

curl -s --max-time 300 http://127.0.0.1:11434/api/generate -d "$PAYLOAD" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])' > "$OUTFILE"
echo "wrote $OUTFILE ($(wc -w < "$OUTFILE") words)"
