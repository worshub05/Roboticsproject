#!/usr/bin/env python3
"""
Scenario suite: one command runs every offline test scenario of the stack and says PASS or FAIL.

Every scenario is DATA: a function that runs it, its parameters (with the range each may take,
used by the falsification search on the next days) and pass criteria on the metrics it returns.
Each check also gives a margin (how far from failing, as a fraction of the threshold); the
scenario's margin is its worst check. Negative margin = FAIL.

Areas
  T  tracking and prediction (Kalman multi-object tracker, lane-following prediction)
  L  lateral control (MPC lane change)
  S  RSS safety distances
  A  emergency braking (AEB)
  V  V2I traffic lights (stop, GLOSA, bad radio, blackout)
  C  CACC platoon (string stability, blackout fallback, cooperative AEB)
  N  negative controls: scenarios that MUST show the problem (they check that a metric can fail)

  python3 tools/scenario_suite.py                 all scenarios
  python3 tools/scenario_suite.py --only C        one area (or one scenario id, e.g. --only V3)
  python3 tools/scenario_suite.py --json report.json
Exit code 0 when everything passes, 1 otherwise (for CI).
"""
import argparse
import importlib
import json
import math
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)


def tool(name):
    """Import one of the offline test scripts in tools/ (they are only run when called directly)."""
    return importlib.import_module(name)


# ----------------------------------------------------------------------------- scenario functions
def mot_two_cars(miss=0.2, noise=0.02, speed=0.3, seed=0):
    """Two cars pass each other in neighbouring lanes; detections noisy and sometimes missed."""
    from lane_follower_perception.mot import MultiObjectTracker
    rnd = random.Random(seed)
    mot = MultiObjectTracker()
    ids, errs = {0: set(), 1: set()}, []
    for k in range(80):                                   # 8 s at 10 Hz
        t = k * 0.1
        truth = [(-2.0 + speed * t, 0.0, speed), (2.0 - speed * t, 0.5, -speed)]
        dets = [dict(x=x + rnd.gauss(0, noise), y=y + rnd.gauss(0, noise), yaw=0.0, length=0.5, width=0.25)
                for x, y, _ in truth if rnd.random() >= miss]
        for tr in mot.step(dets, t):
            car = min(range(2), key=lambda c: math.hypot(tr.x[0] - truth[c][0], tr.x[1] - truth[c][1]))
            if math.hypot(tr.x[0] - truth[car][0], tr.x[1] - truth[car][1]) < 0.3:
                ids[car].add(tr.id)
                if t > 2.0:
                    errs.append(abs(tr.x[2] - truth[car][2]))
    return dict(id_switches=sum(max(0, len(v) - 1) for v in ids.values()),
                cars_tracked=sum(1 for v in ids.values() if v),
                speed_error=float(np.mean(errs)) if errs else math.inf)


def predict_bend(speed=0.3, horizon=3.0, offset=0.05, vel_noise=0.02, seed=0):
    """A car on the bend, a little off its lane centre, with a noisy velocity estimate: the
    lane-following prediction against a straight line (constant velocity)."""
    from lane_follower_perception.predict import predict_cv, predict_lane
    from lane_follower_perception.track_map import L, R, track_path
    rnd = random.Random(seed)
    s0 = L + 0.25 * math.pi * R
    x, y, th = track_path(s0, offset)
    vx = speed * math.cos(th) + rnd.gauss(0, vel_noise)
    vy = speed * math.sin(th) + rnd.gauss(0, vel_noise)
    tx, ty, _ = track_path(s0 + speed * horizon, 0.0)          # it drifts back to the lane centre
    lx, ly = predict_lane(x, y, vx, vy, horizons=(horizon,))[0][0]
    cx, cy = predict_cv(x, y, vx, vy, horizons=(horizon,))[0]
    lane, cv = math.hypot(lx - tx, ly - ty), math.hypot(cx - tx, cy - ty)
    return dict(lane_error=lane, cv_error=cv, improvement=cv / max(lane, 1e-6))


def lane_change(controller='mpc', v=0.5, bend=False, seed=0):
    rms, mx, dw = tool('tune_tracker').run(controller, v, bend, seed)
    return dict(rms_cm=rms, max_cm=mx, steer_change=dw)


def rss_formulas():
    checks = tool('rss_offline_test').formula_checks()
    return dict(failed=sum(1 for _, ok in checks if not ok), total=len(checks))


def rss_case(gap0=3.0, v_ego0=0.7, car_v0=0.0, car_brake_at=None, car_decel=1.5, oncoming=False):
    r = tool('rss_offline_test').run('suite', gap0, v_ego0, car_v0, car_brake_at=car_brake_at,
                                     car_decel=car_decel, oncoming=oncoming, rss=True)
    return dict(collided=int(r['collided']), min_gap=r['min_gap'])


def aeb_case(case='cut-in', gap0=None, rss=True, aeb=True, seed=0):
    r = tool('aeb_offline_test').run(case, rss, aeb, seed=seed, gap0=gap0)
    return dict(collided=int(r['collided']), min_gap=r['min_gap'], aeb_brakes=r['brakes'])


def v2i_cycle(glosa=True, loss=0.0, delay=0.0, blackout_at=None, step=1.0):
    """The car arrives at every moment of the light's cycle (one run per offset)."""
    m = tool('v2i_offline_test')
    rs = [m.run(o, glosa=glosa, loss=loss, delay=delay, blackout_at=blackout_at, seed=i)
          for i, o in enumerate(np.arange(0.0, 25.0, step))]
    return dict(red_violations=sum(r['crossed'] == 'RED' for r in rs), runs=len(rs),
                stops=sum(r['stopped'] for r in rs),
                brake_mJ=1000 * float(np.mean([r['e_brake'] for r in rs])))


def platoon(mode='CACC h=0.3', n=2, loss=0.0, delay=0.1, blackout_at=None, aeb=None, skip=0.0, seed=0):
    r = tool('cacc_offline_test').run(mode, n=n, seed=seed, loss=loss, delay=delay, blackout_at=blackout_at,
                                      aeb=aeb, skip=skip)
    return dict(amplification=r['amp'], min_gap=r['min_gap'], collided=int(r['collided']),
                aeb_brakes=r['aeb_brakes'])


# ----------------------------------------------------------------------------- the scenarios
# checks: (metric, '<=' | '>=' | '==', threshold).  space: parameter ranges for falsification.
SCENARIOS = [
    dict(id='T1', name='tracker: two cars pass, 20% missed detections', fn=mot_two_cars, params=dict(),
         checks=[('id_switches', '==', 0), ('cars_tracked', '==', 2), ('speed_error', '<=', 0.05)],
         space=dict(miss=(0.0, 0.5), noise=(0.0, 0.06), speed=(0.1, 0.6))),
    dict(id='T2', name='prediction: car on the bend, 3 s ahead', fn=predict_bend, params=dict(),
         checks=[('lane_error', '<=', 0.10), ('improvement', '>=', 2.0)],
         space=dict(speed=(0.1, 0.6), horizon=(1.0, 3.0), offset=(-0.15, 0.15), vel_noise=(0.0, 0.08))),
    dict(id='L1', name='MPC lane change, straight, 0.5 m/s', fn=lane_change, params=dict(v=0.5, bend=False),
         checks=[('rms_cm', '<=', 3.0), ('max_cm', '<=', 8.0)], space=dict(v=(0.3, 0.7))),
    dict(id='L2', name='MPC lane change, bend, 0.7 m/s', fn=lane_change, params=dict(v=0.7, bend=True),
         checks=[('rms_cm', '<=', 3.0), ('max_cm', '<=', 8.0)], space=dict(v=(0.3, 0.7))),
    dict(id='S1', name='RSS formula checks', fn=rss_formulas, params=dict(), checks=[('failed', '==', 0)], space={}),
    dict(id='S2', name='RSS: parked car, broken ACC', fn=rss_case, params=dict(gap0=3.0, v_ego0=0.7, car_v0=0.0),
         checks=[('collided', '==', 0), ('min_gap', '>=', 0.10)],
         space=dict(gap0=(0.8, 4.0), v_ego0=(0.3, 0.7))),
    dict(id='S3', name='RSS: tailgating, car in front brakes hard', fn=rss_case,
         params=dict(gap0=0.25, v_ego0=0.5, car_v0=0.5, car_brake_at=2.0),
         checks=[('collided', '==', 0), ('min_gap', '>=', 0.10)],
         space=dict(gap0=(0.15, 1.0), v_ego0=(0.3, 0.7), car_decel=(0.5, 1.5))),
    dict(id='S4', name='RSS: wrong-way car in our lane', fn=rss_case,
         params=dict(gap0=4.0, v_ego0=0.7, car_v0=-0.3, oncoming=True, car_decel=1.0),
         checks=[('collided', '==', 0), ('min_gap', '>=', 0.10)], space=dict(gap0=(1.5, 5.0))),
    dict(id='A1', name='AEB: cut-in 0.25 m ahead + hard brake', fn=aeb_case, params=dict(case='cut-in'),
         checks=[('collided', '==', 0), ('min_gap', '>=', 0.05)], space=dict(gap0=(0.10, 0.6))),
    dict(id='A2', name='AEB: hidden stopped car 0.4 m ahead at 0.7 m/s', fn=aeb_case,
         params=dict(case='late object'), checks=[('collided', '==', 0), ('min_gap', '>=', 0.05)],
         space=dict(gap0=(0.15, 0.8))),
    dict(id='A3', name='AEB: normal ACC following (no false alarm)', fn=aeb_case, params=dict(case='normal ACC'),
         checks=[('aeb_brakes', '==', 0), ('collided', '==', 0)], space={}),
    dict(id='V1', name='V2I stop: every arrival over the cycle', fn=v2i_cycle, params=dict(glosa=False),
         checks=[('red_violations', '==', 0)], space={}),
    dict(id='V2', name='V2I + GLOSA: no red, no stops', fn=v2i_cycle, params=dict(glosa=True),
         checks=[('red_violations', '==', 0), ('stops', '==', 0), ('brake_mJ', '<=', 50.0)], space={}),
    dict(id='V3', name='V2I + GLOSA, 30% lost + 0.3 s delay', fn=v2i_cycle, params=dict(loss=0.3, delay=0.3),
         checks=[('red_violations', '==', 0)], space=dict(loss=(0.0, 0.9), delay=(0.0, 1.0))),
    dict(id='V4', name='V2I blackout 5 m before the light (fail-safe)', fn=v2i_cycle,
         params=dict(blackout_at=10.0), checks=[('red_violations', '==', 0)], space={}),
    dict(id='C1', name='CACC 3 cars: string stable at 0.54 m', fn=platoon, params=dict(n=2),
         checks=[('amplification', '<=', 1.05), ('collided', '==', 0)],
         space=dict(loss=(0.0, 0.5), delay=(0.05, 0.3))),
    dict(id='C2', name='CACC 7 cars: string stable', fn=platoon, params=dict(n=6),
         checks=[('amplification', '<=', 1.05), ('collided', '==', 0)], space={}),
    dict(id='C3', name='CACC: V2V blackout -> ACC fallback', fn=platoon, params=dict(n=2, blackout_at=30.0),
         checks=[('collided', '==', 0), ('min_gap', '>=', 0.30)], space=dict(blackout_at=(15.0, 60.0))),
    dict(id='C4', name='CACC + cooperative AEB, late pose updates', fn=platoon,
         params=dict(n=2, aeb='v2v', skip=0.5), checks=[('aeb_brakes', '==', 0), ('amplification', '<=', 1.05)],
         space=dict(skip=(0.0, 0.8))),
    dict(id='N1', name='negative control: ACC 0.3 s gap is unstable', fn=platoon,
         params=dict(mode='ACC h=0.3', n=6), checks=[('amplification', '>=', 1.10)], space={}),
    dict(id='N2', name='negative control: no V2I runs red lights', fn=lambda: dict(
        red_violations=sum(tool('v2i_offline_test').run(o, v2i=False)['crossed'] == 'RED'
                           for o in np.arange(0.0, 25.0, 1.0))),
         params=dict(), checks=[('red_violations', '>=', 1)], space={}),
    dict(id='N3', name='negative control: LiDAR-only AEB brakes platoon', fn=platoon,
         params=dict(n=2, aeb='lidar', skip=0.5), checks=[('aeb_brakes', '>=', 1)], space={}),
]


# ----------------------------------------------------------------------------- evaluation
def margin(value, op, th):
    """How far from failing (> 0 passes). Fraction of the threshold, or +-1 for exact checks."""
    if op == '==':
        return 1.0 if value == th else -1.0
    scale = abs(th) if th else 1.0
    return (th - value) / scale if op == '<=' else (value - th) / scale


def evaluate(sc, **overrides):
    """Run one scenario (optionally with changed parameters). Returns metrics, check results, margin."""
    params = dict(sc['params'], **overrides)
    t0 = time.perf_counter()
    metrics = sc['fn'](**params)
    results = [(m, op, th, metrics[m], margin(metrics[m], op, th)) for m, op, th in sc['checks']]
    worst = min(r[4] for r in results)
    return dict(id=sc['id'], params=params, metrics=metrics, checks=results, margin=worst,
                passed=worst >= 0, seconds=time.perf_counter() - t0)


def fmt(v):
    return f'{v:.3g}' if isinstance(v, float) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='area letter or scenario id (e.g. C or V3)')
    ap.add_argument('--json', default='', help='write a JSON report to this file')
    args = ap.parse_args()
    chosen = [s for s in SCENARIOS if s['id'].startswith(args.only)]
    if not chosen:
        sys.exit(f'no scenario matches "{args.only}" (ids: {", ".join(s["id"] for s in SCENARIOS)})')
    print(f'Scenario suite: {len(chosen)} scenarios')
    print()
    print('| id | scenario                                         | result  | margin | key results |')
    print('|----|--------------------------------------------------|---------|--------|-------------|')
    report = []
    for sc in chosen:
        r = evaluate(sc)
        report.append(r)
        keys = ', '.join(f'{m} {fmt(v)} ({op} {fmt(th)})' for m, op, th, v, _ in r['checks'])
        print(f'| {sc["id"]:<2} | {sc["name"]:<48} | {"✅ PASS" if r["passed"] else "❌ FAIL"} | '
              f'{r["margin"]:+6.2f} | {keys} |', flush=True)
    n_pass = sum(r['passed'] for r in report)
    print()
    print(f'{n_pass}/{len(report)} passed in {sum(r["seconds"] for r in report):.0f} s')
    if args.json:
        with open(args.json, 'w') as f:
            json.dump([dict(r, checks=[list(c) for c in r['checks']]) for r in report], f, indent=1, default=str)
        print(f'report written to {args.json}')
    sys.exit(0 if n_pass == len(report) else 1)


if __name__ == '__main__':
    main()
