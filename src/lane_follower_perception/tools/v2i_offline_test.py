#!/usr/bin/env python3
"""
Offline test of the V2I traffic light stop (no ROS, no Gazebo).

The car drives at 0.5 m/s towards a traffic light 10 m ahead (green 10 s, amber 3 s, red 12 s).
It arrives at every possible moment of the light's cycle (50 runs, phase shifted by 0.5 s each).
Speed control like the road driver: accelerate 0.4 m/s2, brake up to 1.5 m/s2, 0.1 s actuator lag.
The light broadcasts its state (SPaT) every 0.1 s.

Setups
  no V2I            the car cannot see the light (the camera CNN only sees the lane)
  V2I               every message arrives
  V2I, lossy        30% of messages lost and 0.3 s delay
  V2I, blackout     the messages stop 5 m before the light (fail-safe: treat as red, so the car
                    stops at the line and waits there, because it never hears the light turn green)

  python3 tools/v2i_offline_test.py
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.v2i import LightSchedule, V2IPlanner, parse_spat, spat_message  # noqa: E402

STOP_S, CRUISE = 10.0, 0.5


def run(offset, v2i=True, loss=0.0, delay=0.0, blackout_at=None, seed=0):
    rnd = random.Random(seed)
    sched = LightSchedule(offset=offset)
    planner = V2IPlanner()
    dt, t = 0.005, 0.0
    s, v, v_cmd = 0.0, CRUISE, CRUISE           # s = front bumper position
    queue, next_msg, next_ctrl, limit = [], 0.0, 0.0, -1.0
    res = dict(crossed=None, stop_gap=None, wait=0.0, max_decel=0.0, stopped_on_green=False)
    hist = []
    while t < 60.0 and s < STOP_S + 1.0:
        if t >= next_msg - 1e-9:                # the light broadcasts
            next_msg += 0.1
            if not (blackout_at is not None and s > blackout_at) and rnd.random() >= loss:
                queue.append((t + delay, spat_message([(1, STOP_S, sched)], t)))
        while queue and queue[0][0] <= t:
            planner.receive(parse_spat(queue.pop(0)[1]), t)
        if t >= next_ctrl - 1e-9:               # 20 Hz controller
            next_ctrl += 0.05
            if v2i:
                limit, _ = planner.update(s, v, t)
            target = CRUISE if limit < 0 else min(CRUISE, limit)
            v_cmd = max(0.0, v_cmd + float(np.clip(target - v_cmd, -1.5 * 0.05, 0.4 * 0.05)))
        v += (v_cmd - v) * dt / 0.1
        if s < STOP_S <= s + v * dt and res['crossed'] is None:
            res['crossed'] = sched.state_at(t)[0]
        if v < 0.01 and STOP_S - s < 1.0 and s < STOP_S:
            res['wait'] += dt
            if res['stop_gap'] is None:
                res['stop_gap'] = STOP_S - s
                res['stopped_on_green'] = sched.state_at(t)[0] == 'GREEN'
        hist.append(v)
        if len(hist) > 40:                      # braking measured over 0.2 s
            res['max_decel'] = max(res['max_decel'], (hist.pop(0) - v) / 0.2)
        s += v * dt
        t += dt
    return res


def main():
    offsets = np.arange(0.0, 25.0, 0.5)
    setups = [('no V2I', dict(v2i=False)), ('V2I', dict()), ('V2I, 30% lost + 0.3 s delay', dict(loss=0.3, delay=0.3)),
              ('V2I, blackout 5 m before', dict(blackout_at=STOP_S - 5.0))]
    print(f'{len(offsets)} arrivals spread over the whole light cycle (green 10 s, amber 3 s, red 12 s)')
    print()
    print('| setup                          | red-light violations | amber crossings | stops | stop gap to line (mean, range) | max braking | waiting at the end  |')
    print('|--------------------------------|----------------------|-----------------|-------|--------------------------------|-------------|--------------------|')
    for name, kw in setups:
        rs = [run(o, seed=i, **kw) for i, o in enumerate(offsets)]
        red = sum(r['crossed'] == 'RED' for r in rs)
        amber = sum(r['crossed'] == 'AMBER' for r in rs)
        gaps = [r['stop_gap'] for r in rs if r['stop_gap'] is not None]
        stuck = sum(r['crossed'] is None for r in rs)
        gap_txt = f'{np.mean(gaps) * 100:.1f} cm ({min(gaps) * 100:.1f}-{max(gaps) * 100:.1f})' if gaps else '-'
        print(f'| {name:<30} | {("❌ " if red else "✅ ") + str(red):<20} | {amber:<15} | {len(gaps):<5} | '
              f'{gap_txt:<30} | {max(r["max_decel"] for r in rs):5.2f} m/s2  | {stuck:<18} |')


if __name__ == '__main__':
    main()
