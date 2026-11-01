#!/usr/bin/env python3
"""
Offline test of time-to-collision emergency braking (AEB), next to the RSS layer.

1-D worst cases in our lane. The LiDAR distance is measured every 0.1 s (1 cm noise) and every
safety output reaches the road driver 0.1 s later. Speed limits are applied with the road driver's
limits (brake 1.5 m/s2, AEB full brake 3.0 m/s2) and a 0.1 s actuator lag.

  RSS uses TRACKED cars: a new car is only confirmed on its 3rd scan.
  AEB uses the raw scan: it can react on the 1st scan.

Cases (the speed control / ACC is broken and keeps cruising, except in the last case):
  cut-in       a car cuts in 0.25 m ahead of us (we drive 0.5 m/s, it 0.2 m/s) and brakes hard
  late object  a stopped car is first seen only 0.4 m ahead while we drive 0.7 m/s (it was hidden)
  hard brake   we tailgate at 0.3 m, the car in front brakes hard
  normal ACC   working ACC following a slow car: AEB must NOT brake (false alarm check)

  python3 tools/aeb_offline_test.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.aeb import AEB  # noqa: E402
from lane_follower_perception.rss import RSSParams, check  # noqa: E402

P = RSSParams()


def run(case, rss, aeb_on, seed=0, T=12.0, gap0=None):
    rng = np.random.default_rng(seed)
    dt, t = 0.005, 0.0
    acc_ok = case == 'normal ACC'
    if case == 'cut-in':
        ve, gap, vc, brake_at, cruise = 0.5, 0.25, 0.2, 0.0, 0.5
    elif case == 'late object':
        ve, gap, vc, brake_at, cruise = 0.7, 0.4, 0.0, None, 0.7
    elif case == 'hard brake':
        ve, gap, vc, brake_at, cruise = 0.5, 0.3, 0.5, 1.0, 0.5
    else:                                          # normal ACC behind a slow car
        ve, gap, vc, brake_at, cruise = 0.7, 3.0, 0.15, None, 0.7
    if gap0 is not None:
        gap = gap0
    v_cmd, aeb = ve, AEB()
    rss_lim = aeb_lim = -1.0
    pending, next_scan, scans = [], 0.0, 0
    min_gap, collided, aeb_brakes, prev_state = gap, False, 0, 'OFF'
    prev_gap, lead_est = None, None
    while t < T:
        if gap <= 0.0:
            collided = True
            break
        if t >= next_scan - 1e-9:                  # a LiDAR scan
            next_scan += 0.1
            scans += 1
            d = gap + rng.normal(0, 0.01)
            state, a_lim, _ = aeb.step(d, ve, t)
            if state in ('PARTIAL', 'FULL') and prev_state not in ('PARTIAL', 'FULL'):
                aeb_brakes += 1
            prev_state = state
            r_lim = -1.0
            if scans >= 3:                         # track confirmed on the 3rd scan
                r = check(dict(v_lon=ve, v_lat=0.0), dict(ds=d, dd=-0.275, side=1, v_lon=vc, v_lat=0.0), P)
                r_lim = r['v_limit'] if math.isfinite(r['v_limit']) else -1.0
            # ACC (only when working): constant time gap behind the measured car
            acc_target = cruise
            if acc_ok:
                if prev_gap is not None:
                    raw = ve + (d - prev_gap) / 0.1
                    lead_est = raw if lead_est is None else 0.85 * lead_est + 0.15 * raw
                prev_gap = d
                lead = max(0.0, lead_est if lead_est is not None else 0.0)
                acc_target = float(np.clip(lead + 0.6 * (d - (0.5 + 1.0 * ve)), 0.0, cruise))
            pending.append((t + 0.1, r_lim, a_lim, acc_target))
        while pending and pending[0][0] <= t:
            _, rss_lim, aeb_lim, acc_target = pending.pop(0)
        target, decel = acc_target, 1.5
        if rss and rss_lim >= 0:
            target = min(target, rss_lim)
        if aeb_on and aeb_lim >= 0:
            target = min(target, aeb_lim)
            if aeb_lim == 0.0:
                decel = 3.0                         # emergency brake
        v_cmd = max(0.0, v_cmd + max(-decel * dt, min(0.4 * dt, target - v_cmd)))
        ve += (v_cmd - ve) * dt / 0.1
        if brake_at is not None and t >= brake_at:
            vc = max(0.0, vc - 1.5 * dt)
        gap += (vc - ve) * dt
        min_gap = min(min_gap, gap)
        t += dt
    return dict(collided=collided, min_gap=max(0.0, min_gap), brakes=aeb_brakes)


def main():
    cases = ['cut-in', 'late object', 'hard brake', 'normal ACC']
    setups = [('none', False, False), ('RSS only', True, False), ('AEB only', False, True), ('RSS + AEB', True, True)]
    print(f'| {"case":<12} | {"safety layers":<13} | collisions (5 seeds) | smallest gap (mean) | AEB brake events (5 runs) |')
    print('|--------------|---------------|----------------------|---------------------|---------------------------|')
    for case in cases:
        for name, rss, aeb_on in setups:
            rs = [run(case, rss, aeb_on, seed) for seed in range(5)]
            coll = sum(r['collided'] for r in rs)
            mg = np.mean([r['min_gap'] for r in rs])
            br = sum(r['brakes'] for r in rs)
            print(f'| {case:<12} | {name:<13} | {("❌ " if coll else "✅ ") + str(coll) + " / 5":<20} | '
                  f'{mg:16.2f} m  | {br if aeb_on else "-":>24} |')
    print()
    print('Closest a sudden obstacle can appear and still be avoided (no collision in 5 seeds):')
    print('| case         | RSS only | AEB only | RSS + AEB |')
    print('|--------------|----------|----------|-----------|')
    for case in ('cut-in', 'late object'):
        row = []
        for _, rss, aeb_on in setups[1:]:
            best = None
            for g in np.arange(0.60, 0.04, -0.05):
                if any(run(case, rss, aeb_on, seed, gap0=float(g))['collided'] for seed in range(5)):
                    break
                best = g
            row.append('-' if best is None else f'{best:.2f} m')
        print(f'| {case:<12} | {row[0]:>8} | {row[1]:>8} | {row[2]:>9} |')


if __name__ == '__main__':
    main()
