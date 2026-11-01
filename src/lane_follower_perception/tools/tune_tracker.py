#!/usr/bin/env python3
"""
Fair baseline: tune the simple path tracker's gains and compare it with the MPC.

Same offline closed-loop test as mpc_offline_test.py (kinematic car in track coordinates,
0.1 s actuator lag, sensor noise, the planner's 0.5 m lane change over 1.2 m), on a straight
and on a 5 m bend, at 0.5 and 0.7 m/s, 2 noise seeds each.

  1. default tracker gains (1.5, 2.0)
  2. best tracker that steers no rougher than the MPC (same steering-change budget)
  3. best tracker with no limit on how rough the steering is
  4. the MPC, with its solve times

  python3 tools/tune_tracker.py
"""
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.mpc import LateralMPC  # noqa: E402

R, DT, CTRL_DT, TAU = 5.0, 0.005, 0.05, 0.1
CASES = [(v, b) for v in (0.5, 0.7) for b in (False, True)]
SEEDS = (0, 1)


def smoothstep(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u), 30 * u * u * (1 - u) ** 2


def ref(s, s0=0.5, d_to=0.5, length=1.2):
    f, df = smoothstep((s - s0) / length)
    return d_to * f, math.atan(d_to * df / length)


def run(ctrl, v, bend, seed, gains=(1.5, 2.0), times=None):
    rng = np.random.default_rng(seed)
    kap = 1.0 / R if bend else 0.0
    s = d = psi = w_act = w = t = next_c = 0.0
    mpc = LateralMPC() if ctrl == 'mpc' else None
    w_hist, err = [], []
    while t < 8.0:
        if t >= next_c - 1e-9:
            next_c += CTRL_DT
            dm, pm = d + rng.normal(0, 0.005), psi + rng.normal(0, 0.01)
            if mpc is None:
                d_r, sl = ref(s)
                w = v * kap / max(0.2, 1 - kap * d_r) - gains[0] * (dm - d_r) - gains[1] * (pm - sl)
            else:
                ks = [s + v * mpc.dt * (k + 1) for k in range(mpc.N)]
                rr = [ref(x) for x in ks]
                d_ref = np.array([a for a, _ in rr])
                psi_ref = np.array([b for _, b in rr])
                kp = kap / np.maximum(0.2, 1 - kap * d_ref)
                t0 = time.perf_counter()
                w = mpc.solve(dm, pm, v, d_ref, psi_ref, kp, w_last=w)[0]
                if times is not None:
                    times.append((time.perf_counter() - t0) * 1000)
            w = float(np.clip(w, -1.5, 1.5))
            w_hist.append(w)
        w_act += (w - w_act) * DT / TAU
        sdot = v * math.cos(psi) / (1 - kap * d)
        s += sdot * DT
        d += v * math.sin(psi) * DT
        psi += (w_act - kap * sdot) * DT
        err.append(d - ref(s)[0])
        t += DT
    e = np.array(err)
    dw = np.diff(w_hist) / CTRL_DT
    return np.sqrt(np.mean(e ** 2)) * 100, np.abs(e).max() * 100, np.sqrt(np.mean(dw ** 2))


def score(ctrl, gains=(1.5, 2.0), times=None):
    return np.array([run(ctrl, v, b, sd, gains, times) for v, b in CASES for sd in SEEDS]).mean(0)


def line(name, r):
    return f'| {name:<48} | {r[0]:6.2f} cm | {r[1]:6.2f} cm | {r[2]:5.2f} rad/s2 |'


def main():
    times = []
    mpc = score('mpc', times=times)
    budget = mpc[2] * 1.05
    default = score('tracker', (1.5, 2.0))
    print('Sweeping tracker gains (a few seconds)...')
    results = {}
    for g in itertools.product([0.5, 1, 1.5, 2, 3, 4, 6, 8, 12], [0.5, 1, 2, 3, 4, 6, 8]):
        results[g] = score('tracker', g)
    smooth = {g: r for g, r in results.items() if r[2] <= budget}
    g_fair = min(smooth, key=lambda g: smooth[g][0]) if smooth else None
    g_best = min(results, key=lambda g: results[g][0])
    print()
    print(f'| {"controller":<48} | path RMS  | path max  | steering change |')
    print('|' + '-' * 50 + '|-----------|-----------|-----------------|')
    print(line('tracker, default gains (1.5, 2.0)', default))
    if g_fair:
        print(line(f'tracker, best as smooth as MPC {g_fair}', smooth[g_fair]))
    print(line(f'tracker, best with no smoothness limit {g_best}', results[g_best]))
    print(line('MPC', mpc))
    print()
    print(f'MPC solve time: median {np.median(times):.1f} ms, 99th percentile {np.percentile(times, 99):.1f} ms, '
          f'max {max(times):.1f} ms over {len(times)} solves')
    if g_best:
        print(f'To match the MPC the tracker needs {results[g_best][2] / mpc[2]:.1f}x rougher steering.')


if __name__ == '__main__':
    main()
