#!/usr/bin/env python3
"""
Offline platoon test: ACC against CACC (no ROS, no Gazebo).

A leader drives at 0.45 m/s and twice brakes to 0.15 m/s (0.8 m/s2), waits 3 s and speeds up again.
N followers behind it. Each follower: LiDAR gap every 0.1 s (1 cm noise), controller at 20 Hz,
road driver comfort limits (accelerate 0.4, brake 1.5 m/s2) and a 0.1 s actuator lag.
V2V: every car broadcasts its speed and acceleration at 10 Hz, received 0.1 s later.

Modes
  ACC h=1.0     road driver's ACC as it is (standstill gap 0.5 m, time gap 1.0 s): safe, long gaps
  ACC h=0.3     same law with CACC's short gaps (0.4 m + 0.3 s), but NO V2V
  CACC h=0.3    short gaps with V2V (speed + acceleration of the car in front)

String stability: the speed dip (how far a car slows down) should not GROW from car to car.
amplification = speed dip of the last follower / speed dip of the leader (> 1 = unstable)

  python3 tools/cacc_offline_test.py
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.cacc import ACCLaw, CACCLaw, LeaderProfile, dip, slew  # noqa: E402

CAR_LEN = 0.5


def run(mode, n=2, T=70.0, seed=0, loss=0.0, delay=0.1):
    rnd = random.Random(seed)
    lead = LeaderProfile()
    h, d0 = (1.0, 0.5) if mode == 'ACC h=1.0' else (0.3, 0.4)
    cruise = lead.cruise
    # positions are front bumpers; start every follower at its desired gap
    gaps0 = d0 + h * cruise
    s = [0.0] + [-(i * (gaps0 + CAR_LEN)) for i in range(1, n + 1)]
    v = [cruise] * (n + 1)
    a = [0.0] * (n + 1)
    v_cmd = [cruise] * (n + 1)
    laws = [None] + [ACCLaw(d0=d0, h=h) if mode.startswith('ACC') else CACCLaw(r=d0, h=h) for _ in range(n)]
    meas_gap = [None] * (n + 1)
    v2v = [(cruise, 0.0, -1.0)] * (n + 1)      # what each car last heard from the car in front
    queue = []
    traces = [[] for _ in range(n + 1)]
    min_gap, gap_err, collided = math.inf, [], False
    dt, t, next_scan, next_ctrl, next_msg = 0.005, 0.0, 0.0, 0.0, 0.0
    while t < T:
        lv, la, _ = lead.speed(t)
        v[0], a[0] = lv, la
        gaps = [None] + [s[i - 1] - CAR_LEN - s[i] for i in range(1, n + 1)]
        if t >= next_scan - 1e-9:
            next_scan += 0.1
            meas_gap = [None] + [g + rnd.gauss(0, 0.01) for g in gaps[1:]]
        if t >= next_msg - 1e-9:
            next_msg += 0.1
            for i in range(n):
                if rnd.random() >= loss:
                    queue.append((t + delay, i + 1, (v[i], a[i], t)))
        while queue and queue[0][0] <= t:
            _, j, msg = queue.pop(0)
            v2v[j] = msg
        if t >= next_ctrl - 1e-9:
            next_ctrl += 0.05
            for i in range(1, n + 1):
                if mode.startswith('ACC'):
                    target = laws[i].step(meas_gap[i], v[i], 0.05)
                    a[i] = (target - v[i]) / 0.5            # what it is aiming for (not used by ACC)
                else:
                    vf, af, _ = v2v[i]
                    target, a[i] = laws[i].step(meas_gap[i], v[i], vf, af, 0.05)
                v_cmd[i] = slew(v_cmd[i], target, 0.05)
        for i in range(1, n + 1):
            v[i] += (v_cmd[i] - v[i]) * dt / 0.1
        for i in range(n + 1):
            s[i] += v[i] * dt
            if t > 5.0:
                traces[i].append(v[i])
        if t > 5.0:
            for i in range(1, n + 1):
                min_gap = min(min_gap, gaps[i])
                gap_err.append(gaps[i] - (d0 + h * v[i]))
                if gaps[i] <= 0:
                    collided = True
        t += dt
    dips = [dip(tr, cruise) for tr in traces]
    return dict(dips=dips, amp=dips[-1] / dips[0], min_gap=min_gap, collided=collided,
                gap_rms=float(np.sqrt(np.mean(np.square(gap_err)))), gap_cruise=gaps0)


def main():
    modes = ['ACC h=1.0', 'ACC h=0.3', 'CACC h=0.3']
    for n in (2, 6):
        print(f'Platoon: leader + {n} followers' + ('  (as in Gazebo: leader, simulated car, our robot)' if n == 2 else ''))
        print('| mode        | gap at 0.45 m/s | speed dip per car (leader first), m/s  | amplification | '
              'smallest gap | gap error RMS | collisions |')
        print('|-------------|-----------------|----------------------------------------|---------------|'
              '--------------|---------------|------------|')
        for m in modes:
            rs = [run(m, n=n, seed=k) for k in range(3)]
            r = rs[0]
            dips = ' '.join(f'{d:.2f}' for d in np.mean([x['dips'] for x in rs], axis=0))
            amp = np.mean([x['amp'] for x in rs])
            coll = sum(x['collided'] for x in rs)
            print(f'| {m:<11} | {r["gap_cruise"]:12.2f} m  | {dips:<38} | {amp:10.2f} {"⚠" if amp > 1.05 else "✅"} | '
                  f'{min(x["min_gap"] for x in rs):9.2f} m  | {np.mean([x["gap_rms"] for x in rs]) * 100:9.1f} cm  | '
                  f'{"❌ " + str(coll) if coll else "✅ 0":<10} |')
        print()


if __name__ == '__main__':
    main()
