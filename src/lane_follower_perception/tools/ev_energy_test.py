#!/usr/bin/env python3
"""
Offline test of the EV energy model (lane_follower_perception/ev_energy.py), no ROS, no Gazebo.

  1. physics checks       the model obeys the textbook formulas and conserves energy
  2. driving styles       cruise, gentle and hard stop-and-go, with and without regenerative braking
  3. traffic lights       battery energy over the same 16 m with no V2I, V2I stop, V2I + GLOSA
                          (the V2I offline simulation, 50 arrivals over the light cycle)
  4. platoon              battery energy of every car: ACC 1.0 s, ACC 0.3 s and CACC 0.3 s
                          (the CACC offline simulation)
  5. scale                why stops cost little on a 1:10 car at 0.5 m/s and a lot on a real car

  python3 tools/ev_energy_test.py
"""
import dataclasses
import math
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from lane_follower_perception.ev_energy import G, EnergyMeter, EVParams, trace_energy, wheel_force  # noqa: E402

DT = 0.05


def profile(kind, T=60.0, cruise=0.5):
    """Speed traces (sampled every DT): cruise, stop-and-go (gentle or hard) every 10 s."""
    acc, dec = dict(cruise=(0, 0), gentle=(0.2, 0.3), hard=(0.4, 1.5))[kind]
    out, v = [], cruise
    for k in range(int(T / DT)):
        u = (k * DT) % 10.0
        if kind != 'cruise' and 4.0 <= u < 7.0:
            v = max(0.0, v - dec * DT) if u < 6.0 else 0.0       # brake to a stop, stand 1 s
        elif kind != 'cruise':
            v = min(cruise, v + acc * DT)
        out.append(v)
    return out


def physics_checks():
    """(name, ok) list. Each check compares the model with a hand calculation."""
    p, checks = EVParams(), []
    s = trace_energy([0.5] * 1200, DT, p)
    want = p.c_rr * p.mass * G / p.eta_drive + 0.5 * p.rho * p.cda * 0.25 / p.eta_drive
    checks.append(('cruise: J/m = (C_rr m g + drag) / eta_drive', abs(s['traction_J_per_m'] - want) < 0.01 * want))
    ideal = EVParams(c_rr=0.0, cda=0.0, eta_drive=1.0, eta_regen=1.0, p_regen_max=1e9, v_regen_min=0.0)
    up = [min(0.5, 0.4 * k * DT) for k in range(60)]
    s = trace_energy(up + [0.5] * 20 + up[::-1] + [0.0] * 5, DT, ideal)
    checks.append(('lossless car: speeding up and back down costs nothing', abs(s['traction_J']) < 1e-6))
    checks.append(('lossless car: drive energy = kinetic energy 1/2 m v^2', abs(s['drive_J'] - 0.5 * ideal.mass * 0.25) < 1e-3))
    s = trace_energy(up[::-1] + [0.0] * 5, DT, dataclasses.replace(ideal, regen=False))
    checks.append(('no regen: braking energy all lost in friction brakes', abs(s['friction_J'] - 0.375) < 1e-3
                   and s['regen_J'] == 0.0))
    s_on, s_off = trace_energy(profile('hard'), DT, p), trace_energy(profile('hard'), DT, dataclasses.replace(p, regen=False))
    checks.append(('regen saves energy in stop-and-go', s_on['traction_J'] < s_off['traction_J']))
    checks.append(('regen never recovers more than eta_regen of the braking', s_on['regen_J'] <=
                   p.eta_regen * (s_on['regen_J'] / p.eta_regen + s_on['friction_J']) + 1e-9))
    f1, f2 = wheel_force(1.0, 0.0, p), wheel_force(2.0, 0.0, p)
    checks.append(('air drag grows with v^2', abs((f2 - f1) - 3 * 0.5 * p.rho * p.cda) < 1e-9))
    rnd = random.Random(0)
    clean = trace_energy([0.5] * 1200, DT, p)['traction_J']
    m = EnergyMeter(p)
    for k in range(1200):
        m.step(0.5 + rnd.gauss(0, 0.02), k * DT)
    checks.append(('odometry noise (2 cm/s) changes the energy by < 10%', abs(m.traction() - clean) < 0.1 * clean))
    s = trace_energy([0.0] * 200, DT, dataclasses.replace(p, p_aux=10.0))
    checks.append(('standing still: only the computer uses energy', abs(s['total_J'] - 10.0 * 199 * DT) < 1e-6
                   and s['traction_J'] == 0.0))
    return checks


def light_energy(setup, step=0.5):
    """Mean over arrivals spread over the light cycle: (traction J, J/m, recovered J, time s, stops)."""
    import v2i_offline_test as V
    out, stops = [], 0
    for i, o in enumerate(np.arange(0.0, 25.0, step)):
        m = EnergyMeter(alpha=1.0)
        r = V.run(o, seed=i, meter=m, **setup)
        s = m.summary()
        out.append((s['traction_J'], s['traction_J_per_m'], s['regen_J'], s['time_s']))
        stops += r['stopped']
    return list(np.mean(out, axis=0)) + [stops]


def platoon_energy(mode, n=2, seed=0, **kw):
    """Traction energy per metre (J/m) of every car, leader first."""
    import cacc_offline_test as C
    r = C.run(mode, n=n, seed=seed, **kw)
    sub = 10                                           # 0.005 s simulation -> 0.05 s samples
    return [trace_energy(tr[::sub], r['trace_dt'] * sub)['traction_J_per_m'] for tr in r['traces']]


def main():
    print('1. Physics checks')
    checks = physics_checks()
    for name, ok in checks:
        print(f'   {"✅" if ok else "❌"} {name}')
    print(f'   {sum(ok for _, ok in checks)}/{len(checks)} pass')
    print()
    p = EVParams()
    print(f'2. Driving styles, 60 s, cruise 0.5 m/s, a stop every 10 s (gentle: 0.2 / 0.3 m/s2, hard: 0.4 / 1.5 m/s2)')
    print(f'   car {p.mass} kg, C_rr {p.c_rr}, battery to wheel {p.eta_drive:.0%}, regen {p.eta_regen:.0%}')
    print('| style                          | distance | traction  | per metre | recovered | friction brakes |')
    print('|--------------------------------|----------|-----------|-----------|-----------|-----------------|')
    for kind, label in (('cruise', 'steady cruise'), ('gentle', 'stop-and-go, gentle'),
                        ('hard', 'stop-and-go, hard')):
        for regen in (True, False):
            if kind == 'cruise' and not regen:
                continue
            s = trace_energy(profile(kind), DT, dataclasses.replace(p, regen=regen))
            print(f'| {label + ("" if regen else ", no regen"):<30} | {s["distance_m"]:6.1f} m | {s["traction_J"]:7.2f} J | '
                  f'{s["traction_J_per_m"]:5.3f} J/m | {s["regen_J"]:7.2f} J | {s["friction_J"]:13.2f} J |')
    print()
    print('3. Traffic light 15 m ahead, 50 arrivals over the cycle, energy over the same 16 m')
    print('| setup                          | red-light violations | stops | traction | extra energy from the light |')
    print('|--------------------------------|----------------------|-------|----------|-----------------------------|')
    import v2i_offline_test as V
    base = None
    for label, setup in (('no V2I', dict(v2i=False)), ('V2I stop', dict(glosa=False)), ('V2I + GLOSA', dict()),
                         ('V2I + GLOSA, 30% lost + 0.3 s', dict(loss=0.3, delay=0.3))):
        e, _, _, _, stops = light_energy(setup)
        reds = sum(V.run(o, **setup)['crossed'] == 'RED' for o in np.arange(0.0, 25.0, 0.5))
        base = e if base is None else base
        extra = 'baseline (runs red lights)' if label == 'no V2I' else f'+{(e - base) * 1000:.0f} mJ'
        print(f'| {label:<30} | {reds:<20} | {stops:<5} | {e:6.2f} J | {extra:<27} |')
    print()
    print('4. Platoon, battery energy per metre of every car (leader first), seed 0')
    print('| mode        | J/m per car                                   | last car vs CACC |')
    print('|-------------|-----------------------------------------------|------------------|')
    for n in (2, 6):
        res = {m: platoon_energy(m, n=n) for m in ('ACC h=1.0', 'ACC h=0.3', 'CACC h=0.3')}
        for mode, es in res.items():
            rel = es[-1] / res['CACC h=0.3'][-1] - 1
            print(f'| {mode:<11} | {" ".join(f"{e:.3f}" for e in es):<45} | {rel:+15.1%} |')
        print('|             |                                               |                  |')
    print()
    print('5. Scale: a full stop costs the kinetic energy; in metres of steady driving that is v^2 / (2 C_rr g)')
    for name, v, crr in (('our 1:10 car', 0.5, p.c_rr), ('real car, 50 km/h', 50 / 3.6, 0.01),
                         ('real car, 100 km/h', 100 / 3.6, 0.01)):
        print(f'   {name:<18} {v:5.1f} m/s: one stop = {v * v / (2 * crr * G):7.1f} m of driving')
    print('   So the savings measured here are small in joules, but the same driving style saves far more '
          'on a real car.')


if __name__ == '__main__':
    main()
