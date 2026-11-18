#!/usr/bin/env python3
"""
EV integration study: does the stack save energy, and where does a small robot's energy go?

  1. Gazebo runs      energy of every final experiment (from the run summaries of run_experiments.sh)
  2. comparisons      the same scenario with and without a feature (RSS, AEB, V2I, GLOSA, CACC),
                      traction only and with a 10 W computer running the whole time
  3. regen            how much regenerative braking matters (offline, same model, eta_regen 0-70 %)
  4. full scale       the same driving events on a real electric car (energy per stop, per wait)

  python3 tools/ev_study.py [results/experiments] [--md report.md]
"""
import argparse
import dataclasses
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from lane_follower_perception.ev_energy import G, EVParams, trace_energy  # noqa: E402

P_COMPUTER = 10.0     # W, an embedded GPU computer (Jetson class) plus LiDAR and camera

COMPARISONS = [
    ('Safety with a broken ACC', [('E03', 'RSS off'), ('E04', 'RSS on')]),
    ('Cut-in 0.40 m', [('E05', 'no RSS, no AEB'), ('E06', 'RSS'), ('E07', 'RSS + AEB')]),
    ('Traffic lights', [('E10', 'no V2I (runs red)'), ('E11', 'V2I stop'), ('E12', 'V2I + GLOSA'),
                        ('E13', 'GLOSA, 29 % lost')]),
    ('Platoon', [('E15', 'ACC 1.0 s'), ('E16', 'ACC 0.3 s'), ('E17', 'CACC 0.3 s'), ('E18', 'CACC, bad radio'),
                 ('E19', 'CACC, blackout')]),
    ('Overtaking', [('E01', 'MPC'), ('E02', 'tracker')]),
]

ENERGY_RE = re.compile(r'ENERGY: distance ([\d.]+) m in ([\d.]+) s, traction ([-\d.]+) J .*?recovered by regen '
                       r'([\d.]+) J .*?friction brakes ([\d.]+) J')


def parse(folder):
    """{run id: dict(title, distance, time, traction, regen, friction, text)} from the summaries.
    summary_main_session.txt first, then summary.txt (re-runs replace the main session's runs)."""
    runs = {}
    for name in ('summary_main_session.txt', 'summary.txt'):
        path = os.path.join(folder, name)
        if not os.path.exists(path):
            continue
        for block in open(path, encoding='utf-8').read().split('=== ')[1:]:
            head = block.split('\n', 1)[0]
            rid, title = head.split(':', 1) if ':' in head else (head, '')
            m = ENERGY_RE.search(block)
            if not m:
                continue
            d, t, e, r, f = map(float, m.groups())
            runs[rid.strip()] = dict(title=title.strip(), distance=d, time=t, traction=e, regen=r, friction=f,
                                     text=block)
    return runs


def per_m(r, p_aux=0.0):
    return (r['traction'] + p_aux * r['time']) / max(r['distance'], 1e-9)


def regen_table():
    """Offline: traction J/m for stop-and-go and the platoon, regen efficiency 0 / 50 / 70 %."""
    import ev_energy_test as T
    rows = []
    for eta in (0.0, 0.5, 0.7):
        p = dataclasses.replace(EVParams(), eta_regen=eta, regen=eta > 0)
        hard = trace_energy(T.profile('hard'), T.DT, p)['traction_J_per_m']
        gentle = trace_energy(T.profile('gentle'), T.DT, p)['traction_J_per_m']
        rows.append((eta, gentle, hard))
    return rows


def full_scale():
    """Energy of one stop and one wait, for the 1:10 car and a real electric car, in metres of cruising."""
    cars = [('our 1:10 car', 0.5, 3.0, 0.02, 0.018, 0.6, 0.5, P_COMPUTER),
            ('electric car, 50 km/h', 50 / 3.6, 1800.0, 0.01, 0.6, 0.85, 0.65, 1500.0)]
    out = []
    for name, v, m, crr, cda, eta_d, eta_r, p_aux in cars:
        f_cruise = crr * m * G + 0.5 * 1.2 * cda * v * v
        cruise = f_cruise / eta_d                                   # J per metre from the battery
        ke = 0.5 * m * v * v
        stop_hard = ke / eta_d - eta_r * ke * 0.8                   # brake to 0 (80 % by motor), speed up again
        stop_coast = ke / eta_d - f_cruise * (v * v / (2 * crr * G)) / eta_d   # coast down, no driving saved
        wait = p_aux * 10.0                                         # 10 s at the red light
        out.append((name, v, cruise, max(0.0, stop_hard) / cruise, max(0.0, stop_coast) / cruise, wait / cruise,
                    p_aux / v / cruise))
    return out


def report(runs):
    L = []
    L.append('## 1. Energy of every Gazebo run (traction = without the computer)')
    L.append('')
    L.append('| Run | Scenario | Distance | Time | Traction | Recovered | Friction brakes | With a 10 W computer |')
    L.append('|---|---|---|---|---|---|---|---|')
    for rid in sorted(runs, key=lambda k: (k[0], int(re.sub(r'\D', '', k) or 0))):
        r = runs[rid]
        if rid == 'S0':
            continue
        L.append(f'| {rid} | {r["title"]} | {r["distance"]:.1f} m | {r["time"]:.0f} s | {per_m(r):.2f} J/m | '
                 f'{r["regen"]:.1f} J | {r["friction"]:.1f} J | {per_m(r, P_COMPUTER):.1f} J/m |')
    L.append('')
    L.append('## 2. With and without each feature')
    L.append('')
    L.append('| Comparison | Setup | Traction | vs first | With a 10 W computer | vs first |')
    L.append('|---|---|---|---|---|---|')
    for name, items in COMPARISONS:
        base = None
        for rid, label in items:
            if rid not in runs:
                continue
            r = runs[rid]
            t, a = per_m(r), per_m(r, P_COMPUTER)
            base = base or (t, a)
            L.append(f'| {name} | {rid} {label} | {t:.3f} J/m | {100 * (t / base[0] - 1):+.1f} % | {a:.1f} J/m | '
                     f'{100 * (a / base[1] - 1):+.1f} % |')
            name = ''
    L.append('')
    L.append('## 3. Regenerative braking (offline model, 60 s, a stop every 10 s)')
    L.append('')
    L.append('| Regen efficiency | Gentle stop-and-go | Hard stop-and-go |')
    L.append('|---|---|---|')
    for eta, gentle, hard in regen_table():
        L.append(f'| {eta:.0%} | {gentle:.3f} J/m | {hard:.3f} J/m |')
    L.append('')
    L.append('## 4. The same events at full scale (cost in metres of steady cruising)')
    L.append('')
    L.append('| Vehicle | Speed | Cruising | One hard stop | One gentle (coasting) stop | 10 s wait (computer / auxiliaries) | Computer share at cruise |')
    L.append('|---|---|---|---|---|---|---|')
    for name, v, cruise, hard, coast, wait, share in full_scale():
        L.append(f'| {name} | {v:.1f} m/s | {cruise:.2f} J/m | {hard:.1f} m | {coast:.1f} m | {wait:.0f} m | '
                 f'{100 * share / (1 + share):.0f} % |')
    L.append('')
    L.append('Assumptions: 1:10 car as in ev_energy.py with a 10 W computer; electric car 1800 kg, C_rr 0.01,')
    L.append('CdA 0.6 m2, battery to wheel 85 %, regen 65 % (80 % of a hard stop braked by the motor), 1.5 kW')
    L.append('auxiliaries and self-driving computer. A coasting stop lets rolling resistance use up the speed,')
    L.append('so it costs almost nothing extra in traction; only the time spent waiting costs energy.')
    L.append('')
    L.append('## 5. Findings')
    L.append('')
    L.append('- On the 1:10 car the computer and sensors use about 95 % of the energy at cruise, so energy per')
    L.append('  metre is set mostly by how fast the robot gets there, not by how it brakes.')
    L.append('- Traction energy still shows the effect of the driving style: RSS with a broken ACC uses 11 % less')
    L.append('  (no repeated hard braking), CACC uses about 5 % less than ACC in the platoon.')
    L.append('- V2I stops on this car are gentle enough to coast on rolling resistance, so GLOSA saves no')
    L.append('  measurable traction energy here; its benefit is no stops and no waiting.')
    L.append('- On a real electric car one hard stop from 50 km/h costs about as much as 400 m of driving, so')
    L.append('  every stop that GLOSA avoids, and every braking wave that CACC smooths out, matters far more.')
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder', nargs='?', default=os.path.join(HERE, '..', '..', '..', 'results', 'experiments'))
    ap.add_argument('--md', default='')
    args = ap.parse_args()
    runs = parse(args.folder)
    if not runs:
        sys.exit(f'no run summaries with an ENERGY line found in {args.folder}')
    text = report(runs)
    print(text)
    if args.md:
        with open(args.md, 'w', encoding='utf-8') as f:
            f.write('# EV integration study\n\n' + text + '\n')
        print(f'\nreport written to {args.md}')


if __name__ == '__main__':
    main()
