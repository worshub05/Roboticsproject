"""
V2I (vehicle-to-infrastructure) traffic lights, pure python (no ROS) so it can be tested offline.

Infrastructure side (the traffic light / roadside unit):
  LightSchedule  fixed-time signal: green -> amber -> red, with an offset per light
  spat_message   the broadcast, modelled on the SAE J2735 SPaT (signal phase and timing) and MAP
                 messages: for every light its id, where its stop line is (track position s), the
                 current state and the time until it changes

Vehicle side:
  V2IPlanner     for the next light ahead decides STOP or GO and turns that into a speed limit:
    GREEN    no limit
    AMBER    stop if we can stop before the line braking at most b_max, otherwise go
             (a decision made on amber is kept for that approach: no changing our mind)
    RED      stop (unless we already committed to go on amber)
    UNKNOWN  no message for longer than timeout: fail-safe, treat the light as red
    stopping speed limit  v = sqrt(2 * b * (distance to the line - stop_margin)), with
             b = b_comf if that is enough, otherwise the braking actually needed (<= b_max)
"""
import json
import math

from .track_map import PER

STATES = ('GREEN', 'AMBER', 'RED')


class LightSchedule:
    def __init__(self, green=10.0, amber=3.0, red=12.0, offset=0.0):
        self.green, self.amber, self.red, self.offset = green, amber, red, offset
        self.cycle = green + amber + red

    def state_at(self, t):
        """(state, seconds until it changes, seconds since it started)."""
        u = (t + self.offset) % self.cycle
        if u < self.green:
            return 'GREEN', self.green - u, u
        if u < self.green + self.amber:
            return 'AMBER', self.green + self.amber - u, u - self.green
        return 'RED', self.cycle - u, u - self.green - self.amber


def spat_message(lights, t):
    """lights: list of (id, stop_s, LightSchedule). Returns the broadcast as a JSON string."""
    out = []
    for lid, stop_s, sched in lights:
        state, ttc, _ = sched.state_at(t)
        out.append(dict(id=lid, stop_s=round(stop_s, 3), state=state, time_to_change=round(ttc, 2),
                        green=sched.green, amber=sched.amber, red=sched.red))
    return json.dumps(dict(stamp=round(t, 3), lights=out))


def parse_spat(text):
    try:
        msg = json.loads(text)
        return [l for l in msg.get('lights', []) if l.get('state') in STATES]
    except (ValueError, TypeError, AttributeError):
        return []


def ahead(stop_s, s_front):
    """Distance along the track from our front bumper to a stop line (negative just after it)."""
    d = (stop_s - s_front) % PER
    return d - PER if d > PER - 1.0 else d


class V2IPlanner:
    def __init__(self, b_comf=0.5, b_max=1.0, stop_margin=0.10, lookahead=6.0, timeout=1.0):
        self.b_comf, self.b_max, self.margin = b_comf, b_max, stop_margin
        self.lookahead, self.timeout = lookahead, timeout
        self.map = {}          # id -> stop_s (learned from the messages, kept if they stop)
        self.last = {}         # id -> (state, time_to_change, receive time)
        self.decision = {}     # id -> (STOP | GO, braking used) for the current approach
        self.events = []       # human-readable decisions since the last call

    def receive(self, lights, t):
        for l in lights:
            self.map[l['id']] = float(l['stop_s'])
            self.last[l['id']] = (l['state'], float(l.get('time_to_change', 0.0)), t)

    def state_of(self, lid, t):
        if lid not in self.last or t - self.last[lid][2] > self.timeout:
            return 'UNKNOWN', math.inf
        state, ttc, t_rx = self.last[lid]
        return state, ttc - (t - t_rx)

    def update(self, s_front, v, t):
        """Returns (speed limit or -1, id of the light that limits us or None)."""
        v = max(0.0, v)
        limit, who = math.inf, None
        for lid, stop_s in self.map.items():
            dist = ahead(stop_s, s_front)
            if dist < -0.05 or dist > self.lookahead:
                self.decision.pop(lid, None)        # passed it, or still far away: new approach later
                continue
            state, _ = self.state_of(lid, t)
            room = max(dist - self.margin, 0.0)
            need = v * v / (2 * room) if room > 1e-3 else math.inf
            old = self.decision.get(lid)
            if state == 'GREEN':
                if old is not None and old[0] == 'STOP':
                    self.events.append(f'light {lid} GREEN {dist:.2f} m ahead -> go')
                self.decision.pop(lid, None)
                continue
            if old is None:
                if state == 'AMBER' and need > self.b_max:
                    old = ('GO', 0.0)
                    self.events.append(f'light {lid} AMBER {dist:.2f} m ahead at {v:.2f} m/s: stopping would need '
                                       f'{need:.2f} m/s2 > {self.b_max} -> go')
                else:
                    b = min(self.b_max, max(self.b_comf, need))
                    old = ('STOP', b)
                    why = 'no V2I message: fail-safe' if state == 'UNKNOWN' else state
                    self.events.append(f'light {lid} {why} {dist:.2f} m ahead at {v:.2f} m/s -> stop '
                                       f'(braking {max(need, 0.0) if math.isfinite(need) else 0.0:.2f} m/s2)')
                self.decision[lid] = old
            if old[0] == 'STOP':
                lim = math.sqrt(2 * old[1] * room)
                if lim < limit:
                    limit, who = lim, lid
        return (limit if math.isfinite(limit) else -1.0), who
