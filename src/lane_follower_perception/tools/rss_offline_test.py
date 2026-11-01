#!/usr/bin/env python3
"""
Offline tests of the RSS safety layer (no ROS, no Gazebo).

1. Checks of the safe-distance formulas (they must behave the way the RSS paper says).
2. Closed-loop worst cases where the normal speed control (ACC) is BROKEN and just keeps driving.
   The RSS speed limit is applied the way the road driver applies it: measured every 0.1 s
   (LiDAR / tracker rate) with 0.1 s delay, then the speed follows with the road driver's
   limits (accelerate 0.4 m/s2, brake 1.5 m/s2).

  python3 tools/rss_offline_test.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lane_follower_perception.rss import (EGO_HALF_LENGTH, RSSParams, check, d_lat,  # noqa: E402
                                          d_lon_opposite, d_lon_same, max_safe_speed)

P = RSSParams()
CAR_HALF = 0.25


def formula_checks():
    ok = []
    vs = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    ok.append(('safe distance grows with our speed', all(d_lon_same(a, 0.1, P) < d_lon_same(b, 0.1, P)
                                                         for a, b in zip(vs, vs[1:]))))
    ok.append(('a much faster car in front needs only the margin', abs(d_lon_same(0.2, 2.0, P) - P.lon_margin) < 1e-9))
    ok.append(('opposite: wrong-lane car brakes harder -> shorter distance',
               d_lon_opposite(0.5, 0.3, P.b_min, P.b_correct, P) < d_lon_opposite(0.5, 0.3, P.b_correct, P.b_correct, P)))
    ok.append(('lateral: moving towards each other needs more room', d_lat(0.2, 0.0, P) > d_lat(0.0, 0.0, P) > d_lat(-0.2, 0.0, P)))
    v = max_safe_speed(1.0, lambda x: d_lon_same(x, 0.0, P))
    ok.append(('speed limit keeps exactly the safe distance', abs(d_lon_same(v, 0.0, P) - 1.0) < 1e-4))
    # an oncoming car in ITS lane next to us is not dangerous; the same car in OUR lane is
    side = check(dict(v_lon=0.5, v_lat=0.0), dict(ds=0.5, dd=0.225, side=1, v_lon=-0.3, v_lat=0.0), P)
    ours = check(dict(v_lon=0.5, v_lat=0.0), dict(ds=0.5, dd=-0.275, side=1, v_lon=-0.3, v_lat=0.0), P, car_wrong_lane=True)
    ok.append(('oncoming car in the other lane is safe, in our lane it is dangerous', not side['dangerous'] and ours['dangerous']))
    behind = check(dict(v_lon=0.5, v_lat=0.0), dict(ds=-0.3, dd=-0.275, side=1, v_lon=0.1, v_lat=0.0), P)
    ok.append(('a car behind us never limits us (it is responsible)', math.isinf(behind['v_limit'])))
    return ok


def run(name, gap0, v_ego0, car_v0, car_brake_at=None, car_decel=1.5, oncoming=False, rss=True,
        cruise=0.7, T=15.0):
    """1-D worst case in one lane. car_brake_at: time the car in front starts braking."""
    dt, t = 0.005, 0.0
    xe, ve, v_cmd = 0.0, v_ego0, v_ego0
    xc = gap0 + EGO_HALF_LENGTH + CAR_HALF
    vc = car_v0                                     # signed: + our direction, - towards us
    limit, pending, next_meas = math.inf, [], 0.0
    min_gap, t_danger, collided, danger_seen = math.inf, 0.0, False, False
    while t < T:
        gap = xc - xe - EGO_HALF_LENGTH - CAR_HALF
        min_gap = min(min_gap, gap)
        if gap <= 0.0:
            collided = True
            break
        r = check(dict(v_lon=ve, v_lat=0.0), dict(ds=gap, dd=-0.275, side=1, v_lon=vc, v_lat=0.0), P,
                  car_wrong_lane=oncoming)
        if r['dangerous']:
            t_danger += dt
            danger_seen = True
        if t >= next_meas - 1e-9:                   # tracker rate, result used 0.1 s later
            next_meas += 0.1
            pending.append((t + 0.1, r['v_limit']))
        while pending and pending[0][0] <= t:
            limit = pending.pop(0)[1]
        target = min(cruise, limit) if rss else cruise          # broken ACC: just cruise
        v_cmd = max(0.0, v_cmd + max(-1.5 * dt, min(0.4 * dt, target - v_cmd)))
        ve += (v_cmd - ve) * dt / 0.1                            # 0.1 s actuator lag
        # the other car: brakes when told to, or (oncoming, wrong lane) once the situation is dangerous
        if (car_brake_at is not None and t >= car_brake_at) or (oncoming and danger_seen):
            vc = math.copysign(max(0.0, abs(vc) - car_decel * dt), vc) if vc else 0.0
        xe += ve * dt
        xc += vc * dt
        t += dt
    return dict(name=name, rss=rss, collided=collided, min_gap=min_gap, t_danger=t_danger, v_end=ve)


def main():
    print('1. Formula checks')
    checks = formula_checks()
    for text, ok in checks:
        print(f'   {"✅" if ok else "❌"} {text}')
    cases = [
        ('parked car 3 m ahead, ACC broken', dict(gap0=3.0, v_ego0=0.7, car_v0=0.0)),
        ('slow car (0.1 m/s) 3 m ahead, ACC broken', dict(gap0=3.0, v_ego0=0.7, car_v0=0.1)),
        ('tailgating at 0.5 m/s, car in front brakes hard', dict(gap0=0.25, v_ego0=0.5, car_v0=0.5, car_brake_at=2.0)),
        ('wrong-way car in our lane (0.3 m/s), 4 m', dict(gap0=4.0, v_ego0=0.7, car_v0=-0.3, oncoming=True, car_decel=1.0)),
    ]
    print()
    print('2. Closed-loop worst cases (broken ACC), without and with the RSS speed limit')
    print(f'| {"case":<48} | {"RSS":<3} | collision | smallest gap | time in danger        |')
    print('|' + '-' * 50 + '|-----|-----------|--------------|------------------------|')
    for name, kw in cases:
        for rss in (False, True):
            r = run(name, rss=rss, **kw)
            print(f'| {name:<48} | {"on" if rss else "off":<3} | {"❌ yes" if r["collided"] else "✅ no":<9} | '
                  f'{max(0.0, r["min_gap"]):9.2f} m  | {r["t_danger"]:19.2f} s  |')
    print()
    print('All formula checks passed.' if all(ok for _, ok in checks) else 'SOME FORMULA CHECKS FAILED.')


if __name__ == '__main__':
    main()
