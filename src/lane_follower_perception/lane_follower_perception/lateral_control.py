"""
Steering for the planner's own manoeuvres (peek, lane change, pass, return): either the simple
path tracker or the MPC, plus statistics so the two can be compared in Gazebo.

The road driver scales the steering it receives by (its speed / the requested speed), to keep
the path curvature when it slows down for a car ahead. The MPC plans a yaw rate for the speed
the car is ACTUALLY doing, so it is sent as a curvature times the requested speed, which the
road driver's scaling turns back into the planned yaw rate.

Safety net: if a solve fails (an error or a non-finite answer) or takes longer than max_ms, that
step is steered by the simple path tracker instead, and the fallback is counted.
"""
import math
import time

import numpy as np

from .mpc import LateralMPC


class ManoeuvreStats:
    """Path error and steering smoothness over one manoeuvre."""
    def __init__(self, mpc=None):
        self.mpc = mpc                     # MPCSteering, to report deadline hits and fallbacks
        self.reset()

    def reset(self):
        self.err, self.w, self.ms = [], [], []
        self.start = (self.mpc.deadlines, self.mpc.fallbacks) if self.mpc else (0, 0)

    def add(self, err, w, ms=None):
        self.err.append(err)
        self.w.append(w)
        if ms is not None:
            self.ms.append(ms)

    def summary(self, dt):
        if len(self.err) < 3:
            return 'no data'
        e = np.abs(np.array(self.err)) * 100
        dw = np.diff(np.array(self.w)) / dt
        text = (f'path error RMS {np.sqrt(np.mean(e ** 2)):.1f} cm, max {e.max():.1f} cm, '
                f'steering change RMS {np.sqrt(np.mean(dw ** 2)):.2f} rad/s2')
        if self.ms:
            text += f', MPC solve median {np.median(self.ms):.1f} ms, max {max(self.ms):.1f} ms'
            if self.mpc:
                text += (f', budget hits {self.mpc.deadlines - self.start[0]}, '
                         f'fallbacks {self.mpc.fallbacks - self.start[1]}')
        return text


class MPCSteering:
    def __init__(self, max_ms=40.0, gain_lateral=1.5, gain_heading=2.0, **mpc_args):
        self.mpc = LateralMPC(**mpc_args)
        self.w_last = 0.0
        self.max_ms = max_ms                     # slower than this -> use the tracker for this step
        self.kl, self.kh = gain_lateral, gain_heading
        self.deadlines = 0                       # solves cut short by the budget (plan still usable)
        self.fallbacks = 0                       # steps steered by the tracker instead

    def steer(self, s, d, e_head, v_act, v_cmd, ref_at, curvature_at):
        """
        s, d, e_head  current track coordinates and heading error of the car
        v_act, v_cmd  measured speed and the speed being requested from the road driver
        ref_at(s)     -> (d_ref, slope) of the planned path at track position s
        curvature_at(s) -> curvature of our lane at s
        Returns (yaw rate to publish, solve time in ms).
        """
        m = self.mpc
        v = max(float(v_act), 0.1)
        ks = [s + v * m.dt * (k + 1) for k in range(m.N)]
        refs = [ref_at(x) for x in ks]
        d_ref = np.array([r[0] for r in refs])
        psi_ref = np.array([math.atan(r[1]) for r in refs])
        kap = np.array([curvature_at(x) / max(0.2, 1.0 - curvature_at(x) * r[0]) for x, r in zip(ks, refs)])
        t0 = time.perf_counter()
        try:
            w = float(m.solve(d, e_head, v, d_ref, psi_ref, kap, w_last=self.w_last)[0])
            ok = math.isfinite(w)
        except Exception:
            ok = False
        ms = (time.perf_counter() - t0) * 1000
        if ok and getattr(m, 'solve_info', {}).get('deadline'):
            self.deadlines += 1
        if not ok or ms > self.max_ms:
            # fallback: the planner's simple path tracker, for this step only
            self.fallbacks += 1
            d_r, slope = ref_at(s)
            k0 = curvature_at(s)
            w_fb = float(v_cmd) * k0 / max(0.2, 1.0 - k0 * d_r) - self.kl * (d - d_r) - self.kh * (e_head - math.atan(slope))
            w_fb = float(np.clip(w_fb, -1.5, 1.5))
            self.w_last = w_fb / max(float(v_cmd), 1e-3) * v      # what the car will really turn at
            m.z_prev = np.zeros(m.N)                               # start the next solve fresh
            return w_fb, ms
        self.w_last = w
        w_pub = w / v * max(float(v_cmd), 1e-3)          # curvature x requested speed (see module docstring)
        return float(np.clip(w_pub, -1.5, 1.5)), ms
