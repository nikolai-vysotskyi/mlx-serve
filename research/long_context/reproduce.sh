#!/bin/bash
# Reproduce one arm with the public llmprobe protocol; owns only its server PID.
set -euo pipefail

if [[ $# != 5 && $# != 6 ]]; then
    echo "Usage: $0 /absolute/mlx-serve /absolute/runtime-lib-paths /absolute/gpu_lock.sh on-or-off /absolute/output-directory [model-id]" >&2
    exit 2
fi
binary=$1
runtime=$2
lock=$3
arm=$4
out=$5
[[ "$arm" == on || "$arm" == off ]] || exit 2
port=11234
model=${6:-ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit}
owner="pr408-long-$arm-$$"
server_pid=""
lock_held=0
mkdir -p "$out"
[[ ! -e "$out/server.log" ]] || { echo "Output already exists" >&2; exit 2; }
python3 - "$port" <<'PY'
import socket, sys
with socket.socket() as s:
    s.settimeout(.2)
    if s.connect_ex(('127.0.0.1', int(sys.argv[1]))) == 0:
        raise SystemExit('Port occupied; refusing to replace another server')
PY
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        for _ in {1..100}; do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep .2
        done
        kill -KILL "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
    if [[ "$lock_held" == 1 ]]; then "$lock" release "$owner"; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"$lock" acquire "$owner"
lock_held=1
flag=0
[[ "$arm" == on ]] && flag=1
shasum -a 256 "$binary" > "$out/binary.sha256"
env DYLD_LIBRARY_PATH="$runtime" MLX_SERVE_PREFILL_CHUNK=8192 \
    MLX_SERVE_PREFILL_TRACE=1 QWEN4_PLE_PAR=16 \
    MLX_SERVE_QSA_PAIR="$flag" MLX_SERVE_HC_PREFILL="$flag" \
    MLX_SERVE_GDN_PREFILL_FUSED="$flag" \
    "$binary" serve --host 127.0.0.1 --port "$port" --ctx-size 131072 \
    --prefix-cache-entries 0 --prefill-chunk 8192 > "$out/server.log" 2>&1 &
server_pid=$!
python3 - "$port" "$model" "$out" <<'PY'
import json, socket, sys, time, urllib.request
from pathlib import Path
port, model, out = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
url = f'http://127.0.0.1:{port}'
deadline = time.monotonic() + 30
while True:
    with socket.socket() as s:
        s.settimeout(.2)
        if s.connect_ex(('127.0.0.1', port)) == 0:
            break
    if time.monotonic() > deadline:
        raise SystemExit('Server bind timeout; inspect server.log')
    time.sleep(.2)
body = {'model': model, 'max_tokens': 12, 'stream': False,
        'temperature': 0, 'seed': 1234, 'enable_mtp': False,
        'messages': [{'role': 'user', 'content': 'Capital of France? Answer with one word.'}]}
req = urllib.request.Request(url + '/v1/messages', data=json.dumps(body).encode(),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=300) as r:
    answer = json.load(r)
assert 'paris' in ''.join(c.get('text', '') for c in answer.get('content', [])).lower(), answer
with urllib.request.urlopen(url + '/props') as r:
    props = json.load(r)
(out / 'props.json').write_text(json.dumps(props, indent=2) + '\n')
PY
npx --yes llmprobe@0.6.6 "127.0.0.1:$port" -m "$model" \
    --bench-only --rungs 64k --runs 3 --no-save --no-color \
    --save "$out/llmprobe.json" --label "pr408-long-$arm" | tee "$out/probe.log"
