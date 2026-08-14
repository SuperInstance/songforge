#!/usr/bin/env bash
# Generate lyrics from a local ollama model at a given temperature via API.
# Usage: generate_lyrics.sh <model> <temperature> <outfile> [prompt-file or inline]
set -euo pipefail
MODEL="$1"; TEMP="$2"; OUTFILE="$3"; PROMPT="${4:-}"

if [ -z "$PROMPT" ]; then PROMPT="$(cat /dev/stdin)"; fi

read -r -d '' PAYLOAD <<EOF || true
{"model":"$MODEL","prompt":"$(printf '%s' "$PROMPT" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))' | sed 's/^"//;s/"$//')","stream":false,"options":{"temperature":$TEMP}}
EOF

curl -s http://127.0.0.1:11434/api/generate -d "$PAYLOAD" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["response"])' > "$OUTFILE"
echo "wrote $OUTFILE ($(wc -w < "$OUTFILE") words)"
