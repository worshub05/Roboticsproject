"""
Multi-object tracker: one constant-velocity Kalman filter per car.

State of each track (world frame): [x, y, vx, vy].
Every scan:
  1. predict  - move every track forward by dt with its current velocity (uncertainty grows)
  2. associate - match detections to tracks (gating with the Mahalanobis distance, then
                 nearest-first matching); two tracks can never take the same detection
  3. update   - correct each matched track with its detection
  4. manage   - unmatched detections start TENTATIVE tracks; a track is CONFIRMED after
                3 hits; a track that is not seen keeps being predicted ("coasting") for up
                to max_coast seconds, which carries it through short occlusions and through
                scans where two cars merge into one blob; after that it is deleted.

Pure numpy (no ROS), so it can be tested offline.
"""
import math

import numpy as np

CHI2_GATE = 13.8             # 99.9% gate for a 2-D measurement


class Track:
    def __init__(self, tid, det, t, pos_std, vel_std):
        self.id = tid
        self.x = np.array([det['x'], det['y'], 0.0, 0.0])
        self.P = np.diag([pos_std ** 2, pos_std ** 2, vel_std ** 2, vel_std ** 2])
        self.hits, self.age_start, self.last_seen = 1, t, t
        self.confirmed = False
        self.yaw, self.length, self.width = det['yaw'], det['length'], det['width']

    @property
    def speed(self):
        return math.hypot(self.x[2], self.x[3])


class MultiObjectTracker:
    def __init__(self, meas_std=0.08, accel_std=0.5, init_vel_std=0.6, confirm_hits=3,
                 max_coast=1.0, max_gate_dist=0.7):
        self.R = np.eye(2) * meas_std ** 2
        self.q = accel_std ** 2
        self.init_vel_std = init_vel_std
        self.meas_std = meas_std
        self.confirm_hits = confirm_hits
        self.max_coast = max_coast
        self.max_gate_dist = max_gate_dist
        self.tracks = []
        self.next_id = 1
        self.t = None
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def _predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # white-noise acceleration model
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q
        for tr in self.tracks:
            tr.x = F @ tr.x
            tr.P = F @ tr.P @ F.T + Q

    def step(self, detections, t):
        """detections: list of dicts with x, y (world), yaw, length, width. Returns confirmed tracks."""
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        if dt > 0:
            self._predict(dt)

        # gated costs between every track and every detection
        pairs = []
        for i, tr in enumerate(self.tracks):
            S = self.H @ tr.P @ self.H.T + self.R
            Si = np.linalg.inv(S)
            for j, d in enumerate(detections):
                r = np.array([d['x'], d['y']]) - tr.x[:2]
                m2 = float(r @ Si @ r)
                if m2 < CHI2_GATE and math.hypot(*r) < self.max_gate_dist:
                    # confirmed tracks get first pick, then the closest pairs
                    pairs.append((0 if tr.confirmed else 1, m2, i, j))
        pairs.sort()
        used_t, used_d = set(), set()
        for _, _, i, j in pairs:
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            self._update(self.tracks[i], detections[j], t)

        for j, d in enumerate(detections):
            if j not in used_d:
                self.tracks.append(Track(self.next_id, d, t, self.meas_std, self.init_vel_std))
                self.next_id += 1

        # delete tracks that have not been seen for too long (tentative ones go quickly)
        keep = []
        for tr in self.tracks:
            limit = self.max_coast if tr.confirmed else 0.3
            if t - tr.last_seen <= limit:
                keep.append(tr)
        self.tracks = keep
        return [tr for tr in self.tracks if tr.confirmed]

    def _update(self, tr, d, t):
        z = np.array([d['x'], d['y']])
        S = self.H @ tr.P @ self.H.T + self.R
        K = tr.P @ self.H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - self.H @ tr.x)
        tr.P = (np.eye(4) - K @ self.H) @ tr.P
        tr.hits += 1
        tr.last_seen = t
        tr.yaw, tr.length, tr.width = d['yaw'], d['length'], d['width']
        if tr.hits >= self.confirm_hits:
            tr.confirmed = True
