"""
Lateral model predictive control (MPC) for path tracking in track coordinates.

Model (kinematic unicycle, linearised about the current speed v):
    state   x = [d, psi]   d   = sideways offset from our lane centre (m, + left)
                           psi = heading relative to the lane (rad, + left)
    input   u = w          yaw rate (rad/s)
    d[k+1]   = d[k]   + dt * v * psi[k]
    psi[k+1] = psi[k] + dt * (w[k] - v * kappa[k])          kappa = lane curvature

Cost over a horizon of N steps (dt each):
    sum  q_d * (d - d_ref)^2 + q_psi * (psi - psi_ref)^2       follow the planned path
       + r_w * (w - w_ff)^2                                    stay close to the turn the road needs
       + r_dw * (w[k] - w[k-1])^2 / dt^2                       smooth: penalise changes in yaw rate
    subject to |w| <= w_max and |w[k] - w[k-1]| <= dw_max * dt

The input sequence is solved as a small box-constrained quadratic programme with an
accelerated projected gradient method (FISTA), warm-started from the previous solution.
Pure numpy (no ROS, no solver library), so it can be tested offline.
"""
import numpy as np


class LateralMPC:
    def __init__(self, N=20, dt=0.1, q_d=20.0, q_psi=2.0, r_w=0.05, r_dw=0.02,
                 w_max=1.5, dw_max=4.0, iters=150):
        self.N, self.dt = N, dt
        self.q_d, self.q_psi, self.r_w, self.r_dw = q_d, q_psi, r_w, r_dw
        self.w_max, self.dw_max, self.iters = w_max, dw_max, iters
        self.u_prev = np.zeros(N)

    def _prediction(self, v, kappa):
        """X = Phi x0 + Gam u + drift, stacked [d1, psi1, d2, psi2, ...]."""
        N, dt = self.N, self.dt
        A = np.array([[1.0, dt * v], [0.0, 1.0]])
        B = np.array([0.0, dt])
        Phi = np.zeros((2 * N, 2))
        Gam = np.zeros((2 * N, N))
        drift = np.zeros(2 * N)
        Ak = np.eye(2)
        x_drift = np.zeros(2)
        for k in range(N):
            Ak = A @ Ak
            Phi[2 * k:2 * k + 2] = Ak
            x_drift = A @ x_drift + np.array([0.0, -dt * v * kappa[k]])
            drift[2 * k:2 * k + 2] = x_drift
            # effect of u_j on x_{k+1}: A^(k-j) B
            col = B.copy()
            for j in range(k, -1, -1):
                Gam[2 * k:2 * k + 2, j] = col
                col = A @ col
        return Phi, Gam, drift

    def solve(self, d0, psi0, v, d_ref, psi_ref, kappa, w_last=0.0):
        """
        d_ref, psi_ref, kappa: arrays of length N (values at steps 1..N).
        Returns the planned yaw-rate sequence (length N); apply the first element.
        """
        N, dt = self.N, self.dt
        v = max(float(v), 0.05)
        kappa = np.asarray(kappa, float)
        Phi, Gam, drift = self._prediction(v, kappa)
        xr = np.empty(2 * N)
        xr[0::2], xr[1::2] = d_ref, psi_ref
        Q = np.tile([self.q_d, self.q_psi], N)
        w_ff = v * kappa
        # difference operator for smoothness: (u_k - u_{k-1}), with u_{-1} = w_last
        D = np.eye(N) - np.eye(N, k=-1)
        e0 = np.zeros(N)
        e0[0] = w_last
        s = self.r_dw / dt ** 2
        x_free = Phi @ np.array([d0, psi0]) + drift - xr
        H = Gam.T @ (Q[:, None] * Gam) + self.r_w * np.eye(N) + s * D.T @ D
        f = Gam.T @ (Q * x_free) - self.r_w * w_ff - s * D.T @ e0
        L = float(np.linalg.eigvalsh(H)[-1])
        # box limits on w and (approximately) on its rate, via a running clamp around the previous input
        lo, hi = -self.w_max, self.w_max
        u = np.clip(np.concatenate([self.u_prev[1:], self.u_prev[-1:]]), lo, hi)   # warm start
        y, t = u.copy(), 1.0
        for _ in range(self.iters):
            u_new = np.clip(y - (H @ y + f) / L, lo, hi)
            # rate limit: each step may change by at most dw_max * dt from the one before
            step = self.dw_max * dt
            prev = w_last
            for k in range(N):
                u_new[k] = min(max(u_new[k], prev - step), prev + step)
                prev = u_new[k]
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = u_new + (t - 1) / t_new * (u_new - u)
            u, t = u_new, t_new
        self.u_prev = u
        return u
