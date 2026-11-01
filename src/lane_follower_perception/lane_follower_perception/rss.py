"""
Responsibility-Sensitive Safety (RSS) safe distances (Shalev-Shwartz, Shammah and Shashua, 2017).

RSS turns "drive safely" into checkable rules. Two cars are in a DANGEROUS situation only when
they are too close BOTH along the road (longitudinal) AND sideways (lateral). The safe distances
assume the worst reasonable case: during the response time rho each car may still accelerate,
after that each brakes with at least a minimum deceleration.

  same direction (we are behind a car):
      d_lon = v_r*rho + a_acc*rho^2/2 + (v_r + a_acc*rho)^2 / (2*b_min) - v_f^2 / (2*b_max)
      v_r = our speed, v_f = speed of the car in front; the car in front may brake as hard as
      b_max, we only promise b_min.

  opposite directions (a car coming towards us):
      d_lon = sum over both cars of  (v + v_rho)/2 * rho + v_rho^2 / (2*b)        v_rho = v + a_acc*rho
      the car in the WRONG lane must brake with b_min, the car in its own lane only with the
      gentler b_correct.

  lateral (sideways):
      d_lat = mu + D(u1) + D(u2)
      u = speed of each car TOWARDS the other one; D(u) = how far it moves towards the other
      one if it accelerates sideways with a_lat for rho, then brakes sideways with b_lat.

The proper response when a situation is dangerous: brake (longitudinally) with at least b_min.
Here this is turned into a speed limit: the highest speed at which the gap to every relevant car
is still at least its safe distance.

Pure python/numpy (no ROS), so it can be tested offline.
"""
import math
from dataclasses import dataclass


@dataclass
class RSSParams:
    rho: float = 0.3          # s, response time (LiDAR 10 Hz + tracker + planner + actuator)
    a_acc: float = 0.4        # m/s^2, worst acceleration during the response time
    b_min: float = 1.0        # m/s^2, braking we guarantee (road driver can do 1.5)
    b_max: float = 1.5        # m/s^2, hardest braking we assume the car in front can do
    b_correct: float = 0.7    # m/s^2, braking of a car that is in its own lane (opposite case)
    a_lat: float = 0.3        # m/s^2, worst sideways acceleration during the response time
    b_lat: float = 0.6        # m/s^2, sideways braking after that
    mu: float = 0.05          # m, lateral fluctuation margin
    lon_margin: float = 0.10  # m, extra longitudinal buffer for sensing error
    limit_buffer: float = 0.05  # m, the speed limit aims this far outside the safe distance,
                                # so measurement delay does not push us just inside it


def d_lon_same(v_r, v_f, p: RSSParams):
    """Safe distance to a car in front going the same way."""
    v_r, v_f = max(0.0, v_r), max(0.0, v_f)
    v_rho = v_r + p.a_acc * p.rho
    d = v_r * p.rho + 0.5 * p.a_acc * p.rho ** 2 + v_rho ** 2 / (2 * p.b_min) - v_f ** 2 / (2 * p.b_max)
    return max(0.0, d) + p.lon_margin


def _stop_dist(v, b, p: RSSParams):
    v = max(0.0, v)
    v_rho = v + p.a_acc * p.rho
    return (v + v_rho) / 2 * p.rho + v_rho ** 2 / (2 * b)


def d_lon_opposite(v1, v2, b1, b2, p: RSSParams):
    """Safe distance between two cars driving towards each other (speeds >= 0)."""
    return _stop_dist(v1, b1, p) + _stop_dist(v2, b2, p) + p.lon_margin


def _lat_move(u, p: RSSParams):
    """How far a car moving sideways at u (towards the other car, may be negative) gets towards it."""
    u_rho = u + p.a_lat * p.rho
    return (u + u_rho) / 2 * p.rho + math.copysign(u_rho ** 2 / (2 * p.b_lat), u_rho)


def d_lat(u1, u2, p: RSSParams):
    """Safe sideways distance; u1, u2 = each car's sideways speed towards the other one."""
    return p.mu + max(0.0, _lat_move(u1, p) + _lat_move(u2, p))


def max_safe_speed(gap, d_of_v, v_hi=2.0, iters=30):
    """Highest speed v >= 0 with d_of_v(v) <= gap (d_of_v grows with v); 0 if even standing is unsafe."""
    if d_of_v(0.0) > gap:
        return 0.0
    if d_of_v(v_hi) <= gap:
        return v_hi
    lo, hi = 0.0, v_hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if d_of_v(mid) <= gap else (lo, mid)
    return lo


def check(ego, car, p: RSSParams, ego_wrong_lane=False, car_wrong_lane=False):
    """
    ego, car: dicts in track coordinates, already relative to each other:
        ego: v_lon (our speed along the road, >= 0), v_lat (+ left)
        car: ds (gap along the road between bumpers, + = ahead), dd (sideways gap between the
             sides, >= 0 means apart), side (+1 car is to our left, -1 to our right),
             v_lon (+ = our direction), v_lat (+ left)
    Returns a dict: relevant (car ahead of us), case, lon_gap, lon_safe, lat_gap, lat_safe,
    dangerous, v_limit (our highest safe speed, inf if this car does not limit us).
    """
    out = dict(relevant=car['ds'] > -0.05, case='-', lon_gap=car['ds'], lat_gap=car['dd'],
               lon_safe=0.0, lat_safe=0.0, dangerous=False, v_limit=math.inf)
    if not out['relevant']:
        return out                                   # behind us: the car behind is responsible
    # lateral: speeds towards each other
    u_ego = ego['v_lat'] * car['side']               # we move towards a car on our left if v_lat > 0
    u_car = -car['v_lat'] * car['side']
    out['lat_safe'] = d_lat(u_ego, u_car, p)
    lat_unsafe = car['dd'] < out['lat_safe']
    gap = max(0.0, car['ds'])
    if car['v_lon'] >= -0.02:                        # same direction (or stopped)
        out['case'] = 'same'
        v_f = car['v_lon']
        d_fun = lambda v: d_lon_same(v, v_f, p)      # noqa: E731
    else:                                            # coming towards us
        out['case'] = 'opposite'
        v2 = -car['v_lon']
        if ego_wrong_lane:
            b_ego, b_car = p.b_min, p.b_correct
        elif car_wrong_lane:
            b_ego, b_car = p.b_correct, p.b_min
        else:
            b_ego, b_car = p.b_min, p.b_min
        d_fun = lambda v: d_lon_opposite(v, v2, b_ego, b_car, p)  # noqa: E731
    out['lon_safe'] = d_fun(max(0.0, ego['v_lon']))
    lon_unsafe = gap < out['lon_safe']
    out['dangerous'] = lat_unsafe and lon_unsafe
    if lat_unsafe:
        out['v_limit'] = max_safe_speed(gap - p.limit_buffer, d_fun)
    return out


# ---------------------------------------------------------------- track geometry
EGO_HALF_LENGTH, EGO_HALF_WIDTH = 0.175, 0.15        # our robot, about 0.35 x 0.30 m


def to_track(x, y, yaw=None, v=None, vx=None, vy=None):
    """World pose/velocity -> (s, d, v_lon, v_lat) on the oval track (speeds along / left of our lane)."""
    from .track_map import to_frenet, track_path
    s, d = to_frenet(x, y)
    s, d = float(s[0]), float(d[0])
    th = track_path(s, 0.0)[2]
    if vx is None:
        vx, vy = v * math.cos(yaw), v * math.sin(yaw)
    return s, d, vx * math.cos(th) + vy * math.sin(th), -vx * math.sin(th) + vy * math.cos(th)


def relative(ego_t, car_t, car_half_length=0.25, car_half_width=0.125):
    """
    ego_t, car_t: (s, d, v_lon, v_lat) from to_track. Returns the car dict that check() expects:
    bumper-to-bumper gap along the road and side-to-side gap across it.
    """
    from .track_map import PER, R, on_bend
    s0, d0 = ego_t[0], ego_t[1]
    s1, d1, v_lon, v_lat = car_t
    ds = (s1 - s0 + PER / 2) % PER - PER / 2
    if on_bend(s0):
        ds *= (R - (d0 + d1) / 2) / R                  # s is measured on our lane centre; inside lanes are shorter
    sign = 1.0 if ds >= 0 else -1.0
    return dict(ds=ds - sign * (EGO_HALF_LENGTH + car_half_length),      # < 0 = behind us
                dd=abs(d1 - d0) - EGO_HALF_WIDTH - car_half_width,
                side=1.0 if d1 >= d0 else -1.0, v_lon=v_lon, v_lat=v_lat, d=d1)
