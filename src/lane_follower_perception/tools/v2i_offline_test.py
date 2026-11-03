#!/usr/bin/env python3
"""
Offline test of the V2I traffic light controller (no ROS, no Gazebo).

The car drives at 0.5 m/s towards a traffic light 15 m ahead (green 10 s, amber 3 s, red 12 s).
It arrives at every possible moment of the light's cycle (50 runs, phase shifted by 0.5 s each).
Speed control like the road driver: accelerate 0.4 m/s2, brake up to 1.5 m/s2, 0.1 s actuator lag.
The light broadcasts its state (SPaT) every 0.1 s.

Setups
  no V2I              the car cannot see the light (the camera CNN only sees the lane)
  V2I stop            Day 1: stop at red / amber, go on green
  V2I + GLOSA         Day 2: also slow down early to arrive on green (no stop)
  V2I + GLOSA, lossy  30% of messages lost and 0.3 s delay
  V2I, blackout       the messages stop 5 m before the light (fail-safe: treat as red, so the car
                      stops at the line and waits there, because it never hears the light turn green)

Measured per run: red-light violations, full stops, braking energy (kinetic energy per kg thrown
away by braking, 1/2 * sum of the drops in v^2), and the time to get 1 m past the line.

  python3 tools/v2i_offline_test.py
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.v2i import LightSchedule, V2IPlanner, parse_spat, spat_message  # noqa: E402

START, STOP_S, CRUISE = 0.0, 15.0, 0.5


def run(offset, v2i=True, glosa=True, loss=0.0, delay=0.0, blackout_at=None, seed=0):
    rnd = random.Random(seed)
    sched = LightSchedule(offset=offset)
    planner = V2IPlanner(glosa=glosa)
    dt, t = 0.005, 0.0
    s, v, v_cmd = START, CRUISE, CRUISE          # s = front bumper position
    queue, next_msg, next_ctrl, limit = [], 0.0, 0.0, -1.0
    res = dict(crossed=None, stopped=False, stop_gap=None, e_brake=0.0, t_pass=None, v_line=None)
    while t < 80.0 and s < STOP_S + 1.0:
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
        v_new = v + (v_cmd - v) * dt / 0.1
        res['e_brake'] += max(0.0, 0.5 * (v * v - v_new * v_new))
        v = v_new
        if s < STOP_S <= s + v * dt and res['crossed'] is None:
            res['crossed'], res['v_line'] = sched.state_at(t)[0], v
        if v < 0.01 and s < STOP_S and not res['stopped']:
            res['stopped'], res['stop_gap'] = True, STOP_S - s
        s += v * dt
        t += dt
    res['t_pass'] = t if s >= STOP_S + 1.0 else None
    return res


def main():
    offsets = np.arange(0.0, 25.0, 0.5)
    setups = [('no V2I', dict(v2i=False)),
              ('V2I stop (day 1)', dict(glosa=False)),
              ('V2I + GLOSA (day 2)', dict()),
              ('V2I + GLOSA, 30% lost + 0.3 s', dict(loss=0.3, delay=0.3)),
              ('V2I, blackout 5 m before', dict(blackout_at=STOP_S - 5.0))]
    print(f'{len(offsets)} arrivals spread over the whole light cycle (green 10 s, amber 3 s, red 12 s), '
          f'light {STOP_S:.0f} m ahead, cruise {CRUISE} m/s')
    print()
    print('| setup                          | red-light violations | full stops | braking energy (mean) | '
          'time to pass (mean) | speed at the line (mean) |')
    print('|--------------------------------|----------------------|------------|-----------------------|'
          '---------------------|--------------------------|')
    for name, kw in setups:
        rs = [run(o, seed=i, **kw) for i, o in enumerate(offsets)]
        red = sum(r['crossed'] == 'RED' for r in rs)
        stops = sum(r['stopped'] for r in rs)
        passed = [r['t_pass'] for r in rs if r['t_pass'] is not None]
        tp = f'{np.mean(passed):.1f} s' if passed else 'never (waits)'
        vl = [r['v_line'] for r in rs if r['v_line'] is not None]
        vl_txt = f'{np.mean(vl):.2f} m/s' if vl else '-'
        print(f'| {name:<30} | {("❌ " if red else "✅ ") + str(red):<20} | {stops:>2} / {len(rs):<5} | '
              f'{np.mean([r["e_brake"] for r in rs]) * 1000:14.1f} mJ/kg  | {tp:<19} | {vl_txt:<24} |')


if __name__ == '__main__':
    main()
