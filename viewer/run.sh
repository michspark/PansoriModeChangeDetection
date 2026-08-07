#!/bin/bash
# Pansori Mode Detection Viewer
#
#   ./run.sh                 # serve on http://localhost:7861
#   PORT=8080 ./run.sh       # different port
#   TUNNEL=1 ./run.sh        # also expose publicly via Cloudflare Tunnel
#
# TUNNEL is opt-in: it publishes the viewer (and the audio it serves) to a
# public URL. Set CLOUDFLARED if the binary is not on your PATH.

set -e

PORT=${PORT:-7861}
TUNNEL=${TUNNEL:-0}
CLOUDFLARED=${CLOUDFLARED:-cloudflared}
cd "$(dirname "$0")"

echo "Starting Pansori Mode Detection Viewer at http://localhost:${PORT}"
python app.py --port "$PORT" &
SERVER_PID=$!

TUNNEL_PID=""
if [ "$TUNNEL" = "1" ]; then
    sleep 3
    if ! command -v "$CLOUDFLARED" >/dev/null 2>&1; then
        echo "TUNNEL=1 but '$CLOUDFLARED' not found on PATH. Serving locally only." >&2
    else
        echo "Opening Cloudflare Tunnel..."
        "$CLOUDFLARED" tunnel --url "http://localhost:${PORT}" &
        TUNNEL_PID=$!
    fi
fi

trap 'kill $SERVER_PID $TUNNEL_PID 2>/dev/null; exit' INT TERM

wait $SERVER_PID
