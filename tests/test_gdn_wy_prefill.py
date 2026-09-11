#!/usr/bin/env python3
"""HTTP correctness and engagement guard for the chunkwise GDN WY prefill route.

MLX_SERVE_GDN_WY=1 zig-out/bin/mlx-serve serve \
  --port 18765 --ctx-size 131072 --prefix-cache-entries 0 > /tmp/gdn-wy-server.log 2>&1
python3 tests/test_gdn_wy_prefill.py --log /tmp/gdn-wy-server.log --wy on
Repeat with MLX_SERVE_GDN_WY unset (or =0) and --wy off for the stock arm.
"""
import argparse
import json
import re
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://127.0.0.1:18765')
p.add_argument('--model', default='ddalcu/Qwen3.8-Flash-Next-MLX-Serve-4bit')
p.add_argument('--log', type=Path, required=True)
p.add_argument('--wy', choices=('on', 'off'), required=True)
args = p.parse_args()
log_offset = args.log.stat().st_size if args.log.exists() else 0
source = (Path(__file__).resolve().parents[1] / 'src/transformer.zig').read_text()
corpus = '\n'.join(source.splitlines()[100:900])


def ask(prompt, needle, max_tokens=24):
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'seed': 1234, 'max_tokens': max_tokens,
            'enable_mtp': False, 'enable_pld': False, 'stream': False}
    req = urllib.request.Request(args.url + '/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    usage = result['usage']
    assert usage.get('prompt_tokens_details', {}).get('cached_tokens', 0) == 0, usage
    answer = result['choices'][0]['message']['content']
    needle_ok = needle.lower() in answer.lower()
    print(json.dumps({'wy': args.wy, 'input_tokens': usage.get('prompt_tokens'),
                       'needle_ok': needle_ok}), flush=True)
    return usage, answer, needle_ok


# Request 1: ~13k-token prompt, first passphrase at the very start.
passphrase1 = 'LANTERN-4402'
prompt1 = (f'Remember this passphrase: {passphrase1}.\n' + (corpus * 10)[:60000]
           + '\nWhat passphrase was given at the start? Answer with the passphrase only.')
usage1, answer1, ok1 = ask(prompt1, passphrase1)
assert usage1['prompt_tokens'] > 8192, usage1
assert ok1, answer1

# Request 2: ~6k-token prompt (a DIFFERENT length on purpose -- catches config/shape
# caching bugs between requests), second passphrase.
passphrase2 = 'COBALT-9187'
prompt2 = (f'Remember this passphrase: {passphrase2}.\n' + (corpus * 10)[:28000]
           + '\nWhat passphrase was given at the start? Answer with the passphrase only.')
usage2, answer2, ok2 = ask(prompt2, passphrase2)
assert 2048 < usage2['prompt_tokens'] < usage1['prompt_tokens'], usage2
assert ok2, answer2

# Request 3: short prompt (< 200 tokens), proves the sub-threshold fallback route
# still answers correctly in the same process.
usage3, answer3, ok3 = ask('Capital of France? Answer with one word.', 'paris')
assert usage3['prompt_tokens'] < 200, usage3
assert ok3, answer3

log = args.log.read_bytes()[log_offset:].decode('utf-8', errors='replace')
m = re.search(r'\[gdn-wy\] engaged: S=(\d+)', log)
if args.wy == 'on':
    assert m is not None and int(m.group(1)) >= 256, 'Missing engagement: [gdn-wy] engaged'
else:
    assert m is None, 'Unexpected engagement: [gdn-wy] engaged'
print(f'PASS: long prefill answers correctly and expected wy={args.wy} engagement')
