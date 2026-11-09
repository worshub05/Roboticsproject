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

Day 2 sections
  2. CACC with a worse radio link: lost messages and delay
  3. V2V blackout: the messages stop; CACC falls back to ACC with the long 1.0 s time gap
  4. AEB at short gaps: AEB on the last car, with its closing speed from the LiDAR (differencing
     the distance) or from V2V (cooperative AEB). In Gazebo the other cars are moved at 20 Hz and
     some updates arrive late, so the LiDAR distance jumps by a few cm between scans.

  python3 tools/cacc_offline_test.py
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.aeb import AEB  # noqa: E402
from lane_follower_perception.cacc import ACCLaw, CACCLaw, LeaderProfile, dip, slew  # noqa: E402

CAR_LEN = 0.5


def run(mode, n=2, T=70.0, seed=0, loss=0.0, delay=0.1, blackout_at=None, fallback=True,
        aeb=None, skip=0.0):
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
    backup = [None] + [ACCLaw(d0=0.5, h=1.0) for _ in range(n)]      # fallback when V2V is lost
    meas_gap = [None] * (n + 1)
    v2v = [(cruise, 0.0, 0.0)] * (n + 1)       # what each car last heard from the car in front
    shown = list(s)                            # positions the LiDAR sees (Gazebo pose updates)
    brake, aeb_lim, aeb_brakes, prev_state, next_pose = AEB() if aeb else None, -1.0, 0, 'OFF', 0.0
    fell_back = 0
    queue = []
    traces = [[] for _ in range(n + 1)]
    min_gap, gap_err, collided = math.inf, [], False
    dt, t, next_scan, next_ctrl, next_msg = 0.005, 0.0, 0.0, 0.0, 0.0
    while t < T:
        lv, la, _ = lead.speed(t)
        v[0], a[0] = lv, la
        gaps = [None] + [s[i - 1] - CAR_LEN - s[i] for i in range(1, n + 1)]
        if t >= next_pose - 1e-9:              # cars' poses updated at 20 Hz, some updates late
            next_pose += 0.05
            shown = [x if rnd.random() >= skip else old for x, old in zip(s, shown)]
        if t >= next_scan - 1e-9:
            next_scan += 0.1
            meas_gap = [None] + [shown[i - 1] - CAR_LEN - s[i] + rnd.gauss(0, 0.01) for i in range(1, n + 1)]
            if brake is not None:              # AEB on the last car
                closing = v[n] - v2v[n][0] if aeb == 'v2v' and t - v2v[n][2] < 0.5 else None
                state, aeb_lim, _ = brake.step(meas_gap[n], v[n], t, closing)
                if state in ('PARTIAL', 'FULL') and prev_state not in ('PARTIAL', 'FULL'):
                    aeb_brakes += 1
                prev_state = state
        if t >= next_msg - 1e-9:
            next_msg += 0.1
            for i in range(n):
                if rnd.random() >= loss and (blackout_at is None or t < blackout_at):
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
                elif fallback and t - v2v[i][2] > 0.5 + delay:         # V2V lost: ACC, 1.0 s gap
                    target = backup[i].step(meas_gap[i], v[i], 0.05)
                    laws[i].v_ref = None
                    fell_back = max(fell_back, i)
                else:
                    vf, af, _ = v2v[i]
                    target, a[i] = laws[i].step(meas_gap[i], v[i], vf, af, 0.05)
                    backup[i].step(meas_gap[i], v[i], 0.05)              # keep its speed estimate warm
                if i == n and aeb_lim >= 0:
                    target = min(target, aeb_lim)
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
                gap_rms=float(np.sqrt(np.mean(np.square(gap_err)))), gap_cruise=gaps0,
                aeb_brakes=aeb_brakes, end_gap=gaps[-1], fell_back=fell_back)


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
    day2()


def day2():
    print('2. CACC h=0.3 with a worse radio link (amplification, leader + 6 followers; smallest gap)')
    print('| lost messages | delay 0.1 s       | delay 0.3 s       | delay 0.5 s       |')
    print('|---------------|-------------------|-------------------|-------------------|')
    for loss in (0.0, 0.3, 0.5, 0.8):
        cells = []
        for delay in (0.1, 0.3, 0.5):
            rs = [run('CACC h=0.3', n=6, seed=k, loss=loss, delay=delay, fallback=False) for k in range(3)]
            amp = np.mean([x['amp'] for x in rs])
            cells.append(f'{amp:4.2f} {"✅" if amp <= 1.05 else "⚠"} {min(x["min_gap"] for x in rs):4.2f} m')
        print(f'| {loss:12.0%}  | ' + ' | '.join(f'{c:<17}' for c in cells) + ' |')
    print()
    print('3. V2V blackout at t = 30 s (before the 2nd braking event), leader + 2 followers')
    print('| fallback to ACC (1.0 s gap) | smallest gap | gap at the end | collisions |')
    print('|-----------------------------|--------------|----------------|------------|')
    for fb in (True, False):
        rs = [run('CACC h=0.3', n=2, seed=k, blackout_at=30.0, fallback=fb) for k in range(3)]
        coll = sum(x['collided'] for x in rs)
        print(f'| {"yes" if fb else "no (keeps the last message)":<27} | {min(x["min_gap"] for x in rs):9.2f} m  | '
              f'{np.mean([x["end_gap"] for x in rs]):11.2f} m  | {"❌ " + str(coll) if coll else "✅ 0":<10} |')
    print()
    print('4. AEB on the last car at CACC gaps (0.54 m), leader + 2 followers, 5 seeds')
    print('| late pose updates | AEB closing speed from | AEB brakes (5 runs) | amplification | smallest gap |')
    print('|-------------------|------------------------|---------------------|---------------|--------------|')
    for skip in (0.0, 0.5):
        for src in ('lidar', 'v2v'):
            rs = [run('CACC h=0.3', n=2, seed=k, aeb=src, skip=skip) for k in range(5)]
            amp = np.mean([x['amp'] for x in rs])
            print(f'| {skip:16.0%}  | {"LiDAR (differenced)" if src == "lidar" else "V2V (cooperative)":<22} | '
                  f'{sum(x["aeb_brakes"] for x in rs):<19} | {amp:10.2f} {"✅" if amp <= 1.05 else "⚠"} | '
                  f'{min(x["min_gap"] for x in rs):9.2f} m  |')


if __name__ == '__main__':
    main()
