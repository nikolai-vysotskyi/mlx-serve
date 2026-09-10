#!/usr/bin/env python3
"""Compare completed long-context llmprobe cells; never infer hardware equivalence."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('baseline', type=Path)
p.add_argument('candidate', type=Path)
p.add_argument('--threshold', type=float, default=1.5)
args = p.parse_args()

def cell(path):
    data = json.loads(path.read_text())
    cells = [c for c in data['bench']['contextScaling']
             if c.get('inputTokens', 0) >= 64000 and c.get('prefillTokPerSec')]
    if len(cells) != 1:
        raise ValueError(f'{path}: expected exactly one completed >=64k cell')
    return cells[0]

base, candidate = cell(args.baseline), cell(args.candidate)
length_delta = abs(candidate['inputTokens'] / base['inputTokens'] - 1)
if length_delta > .01:
    raise ValueError('Input lengths differ by >1%; compare matching workloads')
ratio = candidate['prefillTokPerSec'] / base['prefillTokPerSec']
print(json.dumps({
    'baseline_input_tokens': base['inputTokens'],
    'candidate_input_tokens': candidate['inputTokens'],
    'baseline_prefill_tps': base['prefillTokPerSec'],
    'candidate_prefill_tps': candidate['prefillTokPerSec'],
    'throughput_ratio': ratio,
    'exceeds_threshold_in_these_cells': ratio > args.threshold,
    'baseline_runs': base.get('runs'), 'candidate_runs': candidate.get('runs'),
    'note': 'Arithmetic only. Verify hardware, commit/binary identity, cache, engagement, quality and confirmation separately.'
}, indent=2))
