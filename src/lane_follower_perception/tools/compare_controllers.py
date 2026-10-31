#!/usr/bin/env python3
"""
Summarise overtake statistics from saved planner logs (one or more per controller).

  python3 tools/compare_controllers.py ~/ros2_ws/logs/compare_*.log

Reads the lines   ✅ overtake N complete [controller: path error RMS .. cm, max .. cm,
steering change RMS .. rad/s2(, MPC solve median .. ms, max .. ms)]
and prints mean ± standard deviation per controller.
"""
import re
import sys

import numpy as np

PAT = re.compile(r'overtake \d+ complete \[(\w+): path error RMS ([\d.]+) cm, max ([\d.]+) cm, '
                 r'steering change RMS ([\d.]+) rad/s2(?:, MPC solve median ([\d.]+) ms, max ([\d.]+) ms)?\]')
rows = {}
for path in sys.argv[1:]:
    for line in open(path, errors='ignore'):
        m = PAT.search(line)
        if m:
            ctrl = m.group(1)
            vals = [float(x) if x else np.nan for x in m.groups()[1:]]
            rows.setdefault(ctrl, []).append(vals)
if not rows:
    sys.exit('no overtake lines found: check the log file names')
names = ['path error RMS (cm)', 'path error max (cm)', 'steering change RMS (rad/s2)',
         'MPC solve median (ms)', 'MPC solve max (ms)']
ctrls = sorted(rows)
print(f'{"":30s}' + ''.join(f'{c + " (n=" + str(len(rows[c])) + ")":>22s}' for c in ctrls))
for i, n in enumerate(names):
    cells = []
    for c in ctrls:
        v = np.array([r[i] for r in rows[c]])
        v = v[~np.isnan(v)]
        cells.append(f'{v.mean():.2f} ± {v.std():.2f}' if len(v) else '-')
    print(f'{n:30s}' + ''.join(f'{x:>22s}' for x in cells))
