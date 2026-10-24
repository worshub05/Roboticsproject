#!/usr/bin/env python3
"""Generate worlds/track_oval.world: a two-lane oval with a solid edge line, a dashed centre line
and a solid outer edge. Our lane centre runs along y = 0 on the bottom straight (x = 0..20),
turning left around two 5 m-radius bends, so the robot drives anticlockwise in its own lane."""
import math, sys

L, R = 20.0, 5.0                       # straight length, radius of OUR lane centre on the bends
LINES = [(-0.25, 'solid'), (0.25, 'dashed'), (0.75, 'solid')]   # offsets left of our lane centre
ROAD = (-0.45, 0.95)                   # asphalt from right of our lane to left of the other lane
DS = 0.25                              # segment length along the path (m)
DASH_ON, DASH_PERIOD = 0.5, 1.0        # dashed line: 0.5 m paint every 1.0 m

def path(s, off):
    """Point and heading at arc length s along our lane centre, shifted 'off' metres to the left."""
    per = 2 * L + 2 * math.pi * R
    s %= per
    if s < L:                                    # bottom straight, heading +x
        x, y, th = s, 0.0, 0.0
    elif s < L + math.pi * R:                    # right-hand end bend (turning left)
        a = (s - L) / R
        x, y, th = L + R * math.sin(a), R - R * math.cos(a), a
    elif s < 2 * L + math.pi * R:                # top straight, heading -x
        d = s - L - math.pi * R
        x, y, th = L - d, 2 * R, math.pi
    else:                                        # left-hand end bend
        a = (s - 2 * L - math.pi * R) / R
        x, y, th = -R * math.sin(a), R + R * math.cos(a), math.pi + a
    return x - off * math.sin(th), y + off * math.cos(th), th

def seg_len(s, off):
    """Paint length for a segment so curved inner/outer lines stay joined."""
    on_bend = L <= (s % (2 * L + 2 * math.pi * R)) < L + math.pi * R or (s % (2 * L + 2 * math.pi * R)) >= 2 * L + math.pi * R
    return DS * (R - off) / R + 0.02 if on_bend else DS + 0.02

def box(name, x, y, th, sx, sy, z, rgb):
    """One painted rectangle, as a visual inside the single static 'track' model (fast to load)."""
    r, g, b = rgb
    return (f'        <visual name="{name}"><pose>{x:.3f} {y:.3f} {z} 0 0 {th:.4f}</pose>'
            f'<geometry><box><size>{sx:.3f} {sy:.3f} 0.001</size></box></geometry>'
            f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse></material></visual>\n')

out = ['<?xml version="1.0"?>\n<sdf version="1.6">\n  <world name="track_oval">\n'
       '    <physics name="default_physics" default="true" type="ode">\n'
       '      <max_step_size>0.001</max_step_size>\n      <real_time_factor>1.0</real_time_factor>\n    </physics>\n'
       '    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">\n'
       '      <ros><namespace>/gazebo</namespace></ros>\n      <update_rate>1.0</update_rate>\n    </plugin>\n'
       '    <include><uri>model://sun</uri></include>\n    <include><uri>model://ground_plane</uri></include>\n']
out.append('    <model name="track"><static>true</static><link name="surface">\n')
out.append(box('grass', L / 2, R, 0, 80, 60, 0.001, (0.12, 0.28, 0.12)))
per = 2 * L + 2 * math.pi * R
n = int(per / DS)
width = ROAD[1] - ROAD[0]
mid = (ROAD[0] + ROAD[1]) / 2
for i in range(n):                                # asphalt
    s = (i + 0.5) * DS
    x, y, th = path(s, mid)
    out.append(box(f'road_{i}', x, y, th, seg_len(s, mid) + 0.05, width, 0.002, (0.2, 0.2, 0.2)))
for k, (off, style) in enumerate(LINES):          # painted lines
    for i in range(n):
        s = (i + 0.5) * DS
        if style == 'dashed' and (s % DASH_PERIOD) > DASH_ON:
            continue
        x, y, th = path(s, off)
        out.append(box(f'line{k}_{i}', x, y, th, seg_len(s, off), 0.05, 0.003, (1, 1, 1)))
out.append('    </link></model>\n  </world>\n</sdf>\n')
dst = sys.argv[1] if len(sys.argv) > 1 else 'worlds/track_oval.world'
open(dst, 'w').write(''.join(out))
print(f'✅ track written: {dst}  lap {per:.1f} m, {sum(o.count("<visual") for o in out)} painted pieces in 1 model')
