#!/usr/bin/env bash
# Recycle every worker on a RunPod serverless endpoint so jobs stop landing on
# workers still running the previous template image (a template swap alone does
# NOT replace idle/FlashBoot workers).
#
# Usage: RUNPOD_API_KEY=… tools/runpod-recycle-workers.sh <endpoint-id> [timeout-seconds]
#
# Records workersMin/workersMax, sets both to 0, waits until /health reports 0
# workers (default 300 s), then restores the original values.
set -euo pipefail

EP="${1:?usage: $0 <endpoint-id> [timeout-seconds]}"
TIMEOUT="${2:-300}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"

rest() { # method path [json-body]
    local method="$1" path="$2" body="${3:-}"
    if [ -n "$body" ]; then
        curl -sfS -X "$method" "https://rest.runpod.io/v1$path" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" -d "$body"
    else
        curl -sfS -X "$method" "https://rest.runpod.io/v1$path" -H "Authorization: Bearer $RUNPOD_API_KEY"
    fi
}

worker_total() {
    curl -sfS -H "Authorization: Bearer $RUNPOD_API_KEY" "https://api.runpod.ai/v2/$EP/health" \
        | python3 -c 'import sys,json; w=json.load(sys.stdin).get("workers",{}); print(sum(int(v) for v in w.values()))'
}

read -r MIN MAX < <(rest GET "/endpoints/$EP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["workersMin"], d["workersMax"])')
echo "endpoint $EP: workersMin=$MIN workersMax=$MAX, workers now: $(worker_total)"

restore() {
    echo "restoring workersMin=$MIN workersMax=$MAX"
    rest PATCH "/endpoints/$EP" "{\"workersMin\":$MIN,\"workersMax\":$MAX}" >/dev/null
    rest GET "/endpoints/$EP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print("now: workersMin=%s workersMax=%s" % (d["workersMin"], d["workersMax"]))'
}
trap restore EXIT

echo "scaling to 0/0 ..."
rest PATCH "/endpoints/$EP" '{"workersMin":0,"workersMax":0}' >/dev/null

elapsed=0
while :; do
    n="$(worker_total)"
    echo "t=${elapsed}s workers=$n"
    [ "$n" = "0" ] && break
    if [ "$elapsed" -ge "$TIMEOUT" ]; then
        echo "timed out waiting for workers to drain (still $n)" >&2
        exit 1
    fi
    sleep 5; elapsed=$((elapsed + 5))
done
echo "drained — every worker will cold-start on the template's current image"
