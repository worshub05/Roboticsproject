"""
Platooning: ACC (sensor only) against CACC (sensor + V2V), pure python so it can be tested offline.

Spacing policy (both): desired gap = standstill gap + time gap * own speed.
  ACC   only the LiDAR: the speed of the car in front is estimated from how fast the gap changes
        (smoothed, so it lags). This is the road driver's law:
            v_target = v_front_estimate + k_gap * (gap - (d0 + h * v))
        With a long time gap (h = 1.0 s) it is safe; with a short one (h = 0.3 s) every car reacts
        a bit late and a bit harder than the one in front: speed dips GROW down the platoon
        (string instability).
  CACC  the car in front also SENDS its speed and acceleration over V2V (10 Hz), so every car
        reacts at the same time as the one in front:
            a_des = k_a * a_front + k_p * (gap - (r + h * v)) + k_d * (v_front - v)
        the speed command is the integral of a_des. Short gaps (h = 0.3 s) stay string stable.
"""
import json
import math


class ACCLaw:
    def __init__(self, d0=0.5, h=1.0, k_gap=0.6, alpha=0.15, v_max=0.7):
        self.d0, self.h, self.k, self.alpha, self.v_max = d0, h, k_gap, alpha, v_max
        self.prev_gap, self.v_front = None, None

    def step(self, gap, v, dt):
        if self.prev_gap is not None and dt > 1e-3:
            raw = v + (gap - self.prev_gap) / dt
            self.v_front = raw if self.v_front is None else (1 - self.alpha) * self.v_front + self.alpha * raw
        self.prev_gap = gap
        lead = max(0.0, self.v_front if self.v_front is not None else v)
        return min(self.v_max, max(0.0, lead + self.k * (gap - (self.d0 + self.h * v))))


class CACCLaw:
    """
    CACC with graceful degradation: the worse the V2V link (the older the newest message is, which
    grows with delay and with lost messages), the less the car in front's acceleration is trusted
    and the longer the time gap, smoothly from h (good link) up to h_max (as long as ACC's):
        q   = (filtered message age - age_ok) / (age_bad - age_ok), clipped to 0..1
        h       -> h + q * (h_max - h)          (changed at most h_rate seconds per second)
        k_a     -> k_a * (1 - q)
        v_front -> (1 - q) * V2V speed + q * speed measured with the LiDAR (from the gap)
    so with q = 1 nothing from V2V is used any more: a blackout needs no sudden switch.
    The age filter rises fast and recovers slowly, so the gap does not flap on and off.
    """
    def __init__(self, r=0.4, h=0.3, k_a=0.8, k_p=0.6, k_d=2.0, v_max=0.7, a_min=-1.5, a_max=0.4,
                 degrade=True, age_ok=0.15, age_bad=0.45, h_max=1.0, h_rate=0.3):
        self.r, self.h, self.ka, self.kp, self.kd = r, h, k_a, k_p, k_d
        self.v_max, self.a_min, self.a_max = v_max, a_min, a_max
        self.degrade, self.age_ok, self.age_bad, self.h_max, self.h_rate = degrade, age_ok, age_bad, h_max, h_rate
        self.v_ref = None
        self.h_eff, self.age_f, self.q = h, 0.0, 0.0
        self.prev_gap, self.v_lidar = None, None

    def step(self, gap, v, v_front, a_front, dt, age=0.0):
        """age: seconds since the newest V2V message was SENT (delay + time since it arrived)."""
        if self.v_ref is None or abs(self.v_ref - v) > 0.25:
            self.v_ref = v                                   # (re)start from the real speed
        ka = self.ka
        if self.degrade:
            alpha = 0.5 if age > self.age_f else 0.02         # fast when it gets worse, slow to recover
            self.age_f += alpha * (age - self.age_f)
            self.q = min(1.0, max(0.0, (self.age_f - self.age_ok) / (self.age_bad - self.age_ok)))
            target = self.h + self.q * (self.h_max - self.h)
            self.h_eff += min(self.h_rate * dt, max(-self.h_rate * dt, target - self.h_eff))
            ka = self.ka * (1.0 - self.q)
            if self.prev_gap is not None and dt > 1e-3:       # speed of the car in front from the LiDAR gap
                raw = v + (gap - self.prev_gap) / dt
                self.v_lidar = raw if self.v_lidar is None else 0.85 * self.v_lidar + 0.15 * raw
            self.prev_gap = gap
            if self.v_lidar is not None:
                v_front = (1.0 - self.q) * v_front + self.q * max(0.0, self.v_lidar)
        a = ka * a_front + self.kp * (gap - (self.r + self.h_eff * v)) + self.kd * (v_front - v)
        a = min(self.a_max, max(self.a_min, a))
        self.v_ref = min(self.v_max, max(0.0, self.v_ref + a * dt))
        return self.v_ref, a


class LeaderProfile:
    """Cruise, then every period: brake to v_low, hold, accelerate back (a repeatable disturbance)."""
    def __init__(self, cruise=0.45, v_low=0.15, decel=0.8, accel=0.3, hold=3.0, period=25.0, first=12.0):
        self.cruise, self.v_low, self.decel, self.accel = cruise, v_low, decel, accel
        self.hold, self.period, self.first = hold, period, first

    def speed(self, t):
        """(speed, acceleration, event number or 0) at time t."""
        if t < self.first:
            return self.cruise, 0.0, 0
        k, u = divmod(t - self.first, self.period)
        t_down = (self.cruise - self.v_low) / self.decel
        t_up = (self.cruise - self.v_low) / self.accel
        if u < t_down:
            return self.cruise - self.decel * u, -self.decel, int(k) + 1
        if u < t_down + self.hold:
            return self.v_low, 0.0, int(k) + 1
        if u < t_down + self.hold + t_up:
            return self.v_low + self.accel * (u - t_down - self.hold), self.accel, int(k) + 1
        return self.cruise, 0.0, int(k) + 1


def v2v_message(cars, t):
    """cars: list of dicts id, s, v, a. The platoon's V2V broadcast as JSON."""
    return json.dumps(dict(stamp=round(t, 3), cars=[dict(id=c['id'], s=round(c['s'], 3), v=round(c['v'], 4),
                                                         a=round(c['a'], 4)) for c in cars]))


def parse_v2v(text):
    try:
        return {c['id']: c for c in json.loads(text).get('cars', [])}
    except (ValueError, TypeError, AttributeError):
        return {}


def v2v_stamp(text):
    """Time the V2V message was sent (None if it cannot be read)."""
    try:
        return float(json.loads(text)['stamp'])
    except (ValueError, TypeError, KeyError):
        return None


def slew(v_cmd, target, dt, accel=0.4, decel=1.5):
    """Road driver comfort limits on the speed command."""
    return max(0.0, v_cmd + max(-decel * dt, min(accel * dt, target - v_cmd)))


def dip(trace, cruise):
    """How far a speed trace drops below cruise."""
    return max(0.0, cruise - min(trace)) if trace else 0.0
