"""
LiDAR object detection: turn one 2D scan into a list of car-sized objects.

1. Clustering: walk round the scan in beam order; a new cluster starts where two neighbouring
   returns are further apart than a gap that grows with range (beams spread out with distance).
2. Box fitting (L-shape fitting, Zhang et al. 2017): try box orientations from 0 to 90 degrees
   and keep the one where the points hug two edges of the box most closely.
3. Shape prior: the LiDAR only sees the near side(s) of a car, so the visible extent is too small.
   Each box is grown to at least the known car size, AWAY from the sensor, so its centre lands
   on the true centre of the car instead of on the visible face.

Pure numpy (no ROS), so it can be tested offline.
"""
import math

import numpy as np

CAR_LENGTH, CAR_WIDTH = 0.5, 0.25


def cluster_scan(ranges, angle_min, angle_inc, range_min=0.1, range_max=9.8,
                 base_gap=0.12, gap_factor=3.0, min_points=3):
    """Split a scan into clusters. Returns a list of (N, 2) arrays of x, y in the sensor frame."""
    r = np.asarray(ranges, float)
    ang = angle_min + np.arange(len(r)) * angle_inc
    ok = np.isfinite(r) & (r > range_min) & (r < range_max)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    x, y = r[idx] * np.cos(ang[idx]), r[idx] * np.sin(ang[idx])
    gap_limit = base_gap + gap_factor * r[idx] * angle_inc
    jump = np.hypot(np.diff(x), np.diff(y)) > np.maximum(gap_limit[1:], gap_limit[:-1])
    jump |= np.diff(idx) > 2                       # missing beams in between also split
    starts = np.concatenate([[0], np.flatnonzero(jump) + 1])
    ends = np.concatenate([starts[1:], [idx.size]])
    clusters = [np.column_stack([x[a:b], y[a:b]]) for a, b in zip(starts, ends)]
    # the scan is a full circle: join the last cluster to the first if they touch
    if len(clusters) > 1 and idx[0] == 0 and idx[-1] == len(r) - 1:
        a, b = clusters[0][0], clusters[-1][-1]
        if math.hypot(*(a - b)) < max(gap_limit[0], gap_limit[-1]):
            clusters[0] = np.vstack([clusters.pop(), clusters[0]])
    return [c for c in clusters if len(c) >= min_points]


def fit_box(pts, steps=45):
    """
    L-shape fit. Returns centre x, y, yaw, length, width of the tightest rectangle (visible extent).
    Closeness criterion: points should lie near the box edges they are closest to.
    """
    best, best_th = -np.inf, 0.0
    for th in np.linspace(0.0, math.pi / 2, steps, endpoint=False):
        e1 = np.array([math.cos(th), math.sin(th)])
        e2 = np.array([-math.sin(th), math.cos(th)])
        c1, c2 = pts @ e1, pts @ e2
        d1 = np.minimum(c1.max() - c1, c1 - c1.min())
        d2 = np.minimum(c2.max() - c2, c2 - c2.min())
        d = np.maximum(np.minimum(d1, d2), 0.01)
        score = float((1.0 / d).sum())
        if score > best:
            best, best_th = score, th
    e1 = np.array([math.cos(best_th), math.sin(best_th)])
    e2 = np.array([-math.sin(best_th), math.cos(best_th)])
    c1, c2 = pts @ e1, pts @ e2
    l1, l2 = c1.max() - c1.min(), c2.max() - c2.min()
    centre = e1 * (c1.max() + c1.min()) / 2 + e2 * (c2.max() + c2.min()) / 2
    th = best_th
    if l2 > l1:                                    # make the first axis the long one
        l1, l2, th = l2, l1, best_th + math.pi / 2
    return centre[0], centre[1], th, l1, l2


def complete_box(cx, cy, th, length, width, sensor=(0.0, 0.0)):
    """Grow a visible box to the known car size, pushing its centre away from the sensor."""
    e1 = np.array([math.cos(th), math.sin(th)])
    e2 = np.array([-math.sin(th), math.cos(th)])
    c = np.array([cx, cy])
    away = c - np.asarray(sensor, float)
    for axis, size, full in ((e1, length, CAR_LENGTH), (e2, width, CAR_WIDTH)):
        if size < full:
            sign = 1.0 if away @ axis >= 0 else -1.0
            c = c + axis * sign * (full - size) / 2
    return c[0], c[1], th, max(length, CAR_LENGTH), max(width, CAR_WIDTH)


def detect_objects(scan, max_size=1.0):
    """
    One LaserScan-like object -> list of dicts in the SENSOR frame:
    x, y (box centre), yaw (0..pi), length, width, n (points), visible_length, visible_width.
    Clusters bigger than max_size in either direction (walls etc.) are dropped.
    """
    out = []
    for pts in cluster_scan(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min):
        cx, cy, th, l, w = fit_box(pts)
        if l > max_size or w > max_size:
            continue
        if w < 0.06 and l <= 0.35:
            # only one short face seen (the back or front of a car): the car's long axis points
            # straight away from that face
            th, l, w = th + math.pi / 2, w, l
        x, y, th, L, W = complete_box(cx, cy, th, l, w)
        out.append(dict(x=x, y=y, yaw=th % math.pi, length=L, width=W, n=len(pts),
                        visible_length=l, visible_width=w))
    return out
