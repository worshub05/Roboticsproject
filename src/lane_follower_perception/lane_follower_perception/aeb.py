"""
Autonomous emergency braking (AEB) from time-to-collision (TTC), on the raw LiDAR scan.

Why next to RSS: RSS works on TRACKED cars, and a track is only confirmed after 3 scans, so a car
that suddenly appears close in front (cutting in, or seen late) is handled late. AEB looks at
every LiDAR return directly, so it reacts to the very first scan.

1. Path corridor: the points the robot will drive into. The robot's path ahead is an arc with its
   current curvature (yaw rate / speed); a return counts if it lies within half_width of that arc.
   distance = how far along the arc the nearest such return is (from the front bumper).
2. Closing speed: how fast that distance shrinks (measured from scan to scan, smoothed). For a
   return that has just appeared it is assumed to be standing still (closing speed = our speed).
3. TTC = distance / closing speed, then three stages like a real AEB system:
       TTC < warn_ttc   WARNING        (logged only)
       TTC < brake_ttc  PARTIAL brake  speed limit = half our speed
       TTC < full_ttc   FULL brake     speed limit 0, emergency deceleration
   or distance < min_distance: FULL brake whatever the TTC (something is right in front).
   A brake is not released until the TTC has been safe (> release_ttc) for hold_time; a full brake
   always ends in a standstill, which is held until the object is beyond release_distance or has
   moved 0.1 m away (no creeping up to a stopped car).

Pure numpy (no ROS), so it can be tested offline.
"""
import math

import numpy as np


def path_distance(x, y, curvature, half_width=0.2, front=0.15, max_range=4.0):
    """
    x, y: LiDAR points in the robot frame (x forward, y left). Returns the distance along the
    predicted arc from the front bumper to the nearest point inside the corridor (inf if none).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if abs(curvature) < 0.02:
        along, lateral = x, y
    else:
        Rc = 1.0 / curvature                          # centre of the turn at (0, Rc)
        ang = np.arctan2(x, np.abs(Rc) - np.sign(Rc) * y)
        along = np.abs(Rc) * ang
        lateral = np.sign(Rc) * (np.abs(Rc) - np.hypot(x, y - Rc))
    inside = (along > front - 0.02) & (along < max_range) & (np.abs(lateral) < half_width)
    if not inside.any():
        return math.inf
    return max(0.0, float(along[inside].min()) - front)


class AEB:
    STATES = ('OFF', 'WARNING', 'PARTIAL', 'FULL')

    def __init__(self, warn_ttc=1.5, brake_ttc=1.0, full_ttc=0.6, min_distance=0.08,
                 release_ttc=1.5, hold_time=0.3, release_distance=0.3, filter_alpha=0.5, jump=0.25,
                 max_other_speed=0.4):
        self.warn_ttc, self.brake_ttc, self.full_ttc = warn_ttc, brake_ttc, full_ttc
        self.min_distance, self.release_ttc = min_distance, release_ttc
        self.hold_time, self.release_distance = hold_time, release_distance
        self.safe_since = None
        self.d_stop = None
        self.alpha, self.jump = filter_alpha, jump
        self.max_other_speed = max_other_speed      # m/s, fastest any other car drives towards us
        self.prev_d, self.prev_t, self.closing = None, None, None
        self.state = 'OFF'

    def step(self, d, v_ego, t, closing=None):
        """
        d: path distance (inf = nothing), v_ego: our speed, t: time. Returns (state, limit, ttc).
        closing: closing speed known from V2V (our speed - the speed the car in front reports), used
        instead of differencing the LiDAR distance (cooperative AEB). None = LiDAR only.
        """
        v_ego = max(0.0, v_ego)
        # closing speed from the change in distance (new object or big jump: assume it stands still)
        if closing is not None and math.isfinite(d):
            self.closing = closing
        elif math.isfinite(d) and self.prev_d is not None and math.isfinite(self.prev_d) \
                and abs(d - self.prev_d) < self.jump and t > self.prev_t:
            raw = (self.prev_d - d) / (t - self.prev_t)
            self.closing = raw if self.closing is None else (1 - self.alpha) * self.closing + self.alpha * raw
        elif math.isfinite(d):
            self.closing = v_ego
        else:
            self.closing = None
        self.prev_d, self.prev_t = d, t
        closing = self.closing if self.closing is not None else 0.0
        # physically possible closing speed: our speed + the fastest other car. While we steer, the
        # predicted arc swings across the scan and the distance can jump; this stops that from
        # looking like a car rushing at us (false alarms when pulling out past a slow car)
        closing = min(closing, v_ego + self.max_other_speed)
        ttc = d / closing if (math.isfinite(d) and closing > 0.02) else math.inf

        if d < self.min_distance or ttc < self.full_ttc:
            new = 'FULL'
        elif ttc < self.brake_ttc:
            new = 'PARTIAL'
        elif ttc < self.warn_ttc:
            new = 'WARNING'
        else:
            new = 'OFF'
        # hysteresis: keep the current brake level until the TTC has been safe for hold_time
        order = self.STATES.index
        if self.state in ('PARTIAL', 'FULL') and order(new) < order(self.state):
            if ttc > self.release_ttc:
                self.safe_since = self.safe_since if self.safe_since is not None else t
            else:
                self.safe_since = None
            if self.safe_since is None or t - self.safe_since < self.hold_time:
                new = self.state
        else:
            self.safe_since = None
        if self.state == 'PARTIAL' and new == 'PARTIAL' and v_ego < 0.05 and d < self.release_distance:
            new = 'FULL'                                  # an AEB brake ends in a standstill
        if self.state == 'FULL':
            if v_ego <= 0.02 and self.d_stop is None:
                self.d_stop = d                           # where we stopped
            gone = d > self.release_distance or (self.d_stop is not None and d > self.d_stop + 0.1)
            if new != 'FULL' and (v_ego > 0.02 or not gone):
                new = 'FULL'                              # stay stopped until it has moved away
        if new != 'FULL':
            self.d_stop = None
        self.state = new
        limit = {'OFF': -1.0, 'WARNING': -1.0, 'PARTIAL': 0.5 * v_ego, 'FULL': 0.0}[new]
        return new, limit, ttc
