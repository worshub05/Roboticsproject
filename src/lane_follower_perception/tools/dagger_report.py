#!/usr/bin/env python3
"""Table of the DAgger runs in report.jsonl (written by dagger.py).   python3 tools/dagger_report.py [report.jsonl]"""
import json
import os
import sys

path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ros2_ws/dagger/report.jsonl')
rows = [json.loads(line) for line in open(path) if line.strip()]
print('## DAgger: network driving on its own (evaluation runs)\n')
print('| Model | Lane | Steering push | Distance | Lane offset RMS | Max offset | Interventions per 100 m | Expert share | Network vs expert |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    if r['collected'] == 0:
        push = 'none' if r['noise_std'] == 0 else f'{r["noise_std"]} rad/s'
        name = 'expert (beta 1)' if r['beta'] >= 1.0 else r['model']
        print(f'| {name} | {r["lane"]} | {push} | {r["distance_m"]} m | {r["rms_offset_cm"]} cm | {r["max_offset_cm"]} cm | '
              f'{r["interventions_per_100m"]} | {100 * r["expert_share"]:.0f} % | {r["disagreement"]} rad/s |')
print('\n## Data collection rounds\n')
print('| Round | Model driving | beta (expert share of the command) | Lane | Images | Interventions | Lane offset RMS |')
print('|---|---|---|---|---|---|---|')
for r in rows:
    if r['collected'] > 0:
        print(f'| {r["tag"]} | {r["model"]} | {r["beta"]} | {r["lane"]} | {r["collected"]} | {r["interventions"]} | {r["rms_offset_cm"]} cm |')
