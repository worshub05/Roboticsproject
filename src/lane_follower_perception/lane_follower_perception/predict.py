"""
Motion prediction for tracked cars: where will each car be in 1, 2, 3 seconds?

Two predictors, compared in the thesis:
  constant velocity  p(t) = p0 + v * t                (straight line, knows nothing about the road)
  lane following     the car keeps its lane and its speed along the lane: it is placed on the
                     lane centre line at its current sideways offset, then moved along the lane
                     by speed * t. On a bend it follows the curve instead of running off it.

The lane-following prediction also says WHICH lane the car is in and which way it is going,
which the planner can use directly (an oncoming car = other lane, moving against us).
"""
import math

from .track_map import LANE_WIDTH, R, on_bend, to_frenet, track_path

HORIZONS = (1.0, 2.0, 3.0)


def predict_cv(x, y, vx, vy, horizons=HORIZONS):
    return [(x + vx * t, y + vy * t) for t in horizons]


def lane_of(d):
    """0 = our lane, 1 = the other lane, -1 = off the road."""
    if abs(d) < LANE_WIDTH / 2 + 0.1:
        return 0
    if abs(d - LANE_WIDTH) < LANE_WIDTH / 2 + 0.1:
        return 1
    return -1


def predict_lane(x, y, vx, vy, horizons=HORIZONS):
    """Returns (list of predicted (x, y), lane, direction): direction +1 = our way round, -1 = oncoming."""
    s, d = to_frenet(x, y)
    s, d = float(s[0]), float(d[0])
    lane = lane_of(d)
    if lane == -1:                                   # off the road: no lane to follow
        return predict_cv(x, y, vx, vy, horizons), lane, 0
    d_lane = 0.0 if lane == 0 else LANE_WIDTH
    th = track_path(s, d_lane)[2]
    v_along = vx * math.cos(th) + vy * math.sin(th)  # signed speed along our direction of travel
    direction = 1 if v_along >= 0 else -1
    out = []
    for t in horizons:
        # move along the lane's own centre line: on a bend a lane to the inside (d > 0) is shorter,
        # so a distance travelled there corresponds to more of our s
        dist, s_t = v_along * t, s
        step = 0.1 if dist >= 0 else -0.1
        remaining = abs(dist)
        while remaining > 1e-9:
            ds = min(abs(step), remaining)
            scale = R / (R - d_lane) if on_bend(s_t) else 1.0
            s_t += math.copysign(ds * scale, step)
            remaining -= ds
        # drift back towards the lane centre (cars keep to the middle of their lane)
        px, py, _ = track_path(s_t, d_lane + (d - d_lane) * math.exp(-t / 1.5))
        out.append((px, py))
    return out, lane, direction
