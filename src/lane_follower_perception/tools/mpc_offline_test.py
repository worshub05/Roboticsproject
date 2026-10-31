#!/usr/bin/env python3
"""
Offline closed-loop test: lane change with the current path tracker vs the MPC.

Simulates the car as a kinematic unicycle in track coordinates with a 0.1 s actuator lag and
sensor noise, doing the overtake planner's lane change (0.5 m sideways over 1.2 m), on a
straight and on a 5 m bend, at 0.5 and 0.7 m/s, 5 noise seeds each.

  python3 tools/mpc_offline_test.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser('~/ros2_ws/src/lane_follower_perception'))
from lane_follower_perception.mpc import LateralMPC  # noqa: E402
R = 5.0
def smooth(u):
    u = min(max(u, 0.0), 1.0); return u**3*(10-15*u+6*u*u), 30*u*u*(1-u)**2
def ref(s, s0, d_from, d_to, length):
    f, df = smooth((s - s0) / length)
    return d_from + (d_to - d_from) * f, math.atan((d_to - d_from) * df / length)
def run(ctrl, v=0.5, bend=False, length=1.2, d_to=0.5, T=8.0, seed=0, tau=0.1, ctrl_dt=0.05):
    rng = np.random.default_rng(seed)
    kap = (1.0 / R) if bend else 0.0
    s, d, psi, w_act = 0.0, 0.0, 0.0, 0.0
    mpc = LateralMPC()
    w_cmd, w_hist, t, err, rows = 0.0, [], 0.0, [], []
    next_ctrl = 0.0
    dt = 0.005
    while t < T:
        if t >= next_ctrl - 1e-9:
            next_ctrl += ctrl_dt
            dm, pm = d + rng.normal(0, 0.005), psi + rng.normal(0, 0.01)
            k_path = kap / max(0.2, 1 - kap * ref(s, 0.5, 0, d_to, length)[0])
            if ctrl == 'tracker':
                d_r, sl = ref(s, 0.5, 0.0, d_to, length)
                w_cmd = v * k_path - 1.5 * (dm - d_r) - 2.0 * (pm - sl)
            else:
                ks = [s + v * mpc.dt * (k + 1) for k in range(mpc.N)]
                rr = [ref(x, 0.5, 0.0, d_to, length) for x in ks]
                d_ref = np.array([a for a, _ in rr]); psi_ref = np.array([b for _, b in rr])
                kp = np.array([kap / max(0.2, 1 - kap * a) for a in d_ref])
                w_cmd = mpc.solve(dm, pm, v, d_ref, psi_ref, kp, w_last=w_cmd)[0]
            w_cmd = float(np.clip(w_cmd, -1.5, 1.5))
            w_hist.append(w_cmd)
        w_act += (w_cmd - w_act) * dt / tau
        sdot = v * math.cos(psi) / (1 - kap * d)
        s += sdot * dt; d += v * math.sin(psi) * dt; psi += (w_act - kap * sdot) * dt
        d_r = ref(s, 0.5, 0.0, d_to, length)[0]
        err.append(d - d_r); rows.append((t, d))
        t += dt
    err = np.array(err); w = np.array(w_hist)
    ds = [r[1] for r in rows]
    over = max(0.0, max(ds) - d_to)
    settle = next((tt for tt, dd in rows[::-1] if abs(dd - d_to) > 0.03), 0.0)
    return dict(rms=np.sqrt(np.mean(err**2))*100, maxerr=np.abs(err).max()*100, over=over*100,
                w_peak=np.abs(w).max(), dw_peak=np.abs(np.diff(w)).max()/ctrl_dt, w_rms_rate=np.sqrt(np.mean((np.diff(w)/ctrl_dt)**2)), settle=settle)
if __name__ == '__main__':
    for bend in (False, True):
        for v in (0.5, 0.7):
            for ctrl in ('tracker', 'mpc'):
                r = [run(ctrl, v=v, bend=bend, seed=s) for s in range(5)]
                m = {k: np.mean([x[k] for x in r]) for k in r[0]}
                print(f"{'bend' if bend else 'straight':8s} v={v} {ctrl:7s}: lateral error RMS {m['rms']:.1f} cm, max {m['maxerr']:.1f} cm, "
                      f"overshoot {m['over']:.1f} cm, settled at {m['settle']:.1f} s, peak yaw rate {m['w_peak']:.2f} rad/s, "
                      f"yaw-rate change RMS {m['w_rms_rate']:.2f} rad/s2, peak {m['dw_peak']:.1f}")
