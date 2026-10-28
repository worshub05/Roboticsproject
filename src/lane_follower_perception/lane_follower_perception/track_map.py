"""
The oval track as a map (pure maths, no ROS): lane centre lines and track coordinates.

  s  distance along OUR lane centre from the start line (0 .. PER)
  d  sideways offset to the left of our lane centre (the other lane's centre is d = 0.5)
"""
import math

import numpy as np

L, R = 20.0, 5.0                     # must match tools/make_track.py
PER = 2 * L + 2 * math.pi * R
LANE_WIDTH = 0.5


def track_path(s, off=0.0):
    """Point and heading at arc length s along our lane centre, shifted 'off' metres to the left."""
    s %= PER
    if s < L:
        x, y, th = s, 0.0, 0.0
    elif s < L + math.pi * R:
        a = (s - L) / R
        x, y, th = L + R * math.sin(a), R - R * math.cos(a), a
    elif s < 2 * L + math.pi * R:
        d = s - L - math.pi * R
        x, y, th = L - d, 2 * R, math.pi
    else:
        a = (s - 2 * L - math.pi * R) / R
        x, y, th = -R * math.sin(a), R + R * math.cos(a), math.pi + a
    return x - off * math.sin(th), y + off * math.cos(th), th


def to_frenet(x, y):
    """Track coordinates (s, d) for arrays of points."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    s, d = np.empty_like(x), np.empty_like(x)
    bottom = (x >= 0) & (x <= L) & (y < R)
    top = (x >= 0) & (x <= L) & (y >= R)
    right, left = x > L, x < 0
    s[bottom], d[bottom] = x[bottom], y[bottom]
    s[top], d[top] = L + math.pi * R + (L - x[top]), 2 * R - y[top]
    a = np.arctan2(x[right] - L, -(y[right] - R))
    s[right], d[right] = L + R * a, R - np.hypot(x[right] - L, y[right] - R)
    a = np.arctan2(-x[left], y[left] - R)
    s[left], d[left] = 2 * L + math.pi * R + R * a, R - np.hypot(x[left], y[left] - R)
    return s % PER, d


def on_bend(s):
    s %= PER
    return L <= s < L + math.pi * R or s >= 2 * L + math.pi * R
