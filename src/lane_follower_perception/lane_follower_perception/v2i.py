"""
V2I (vehicle-to-infrastructure) traffic lights, pure python (no ROS) so it can be tested offline.

Infrastructure side (the traffic light / roadside unit):
  LightSchedule  fixed-time signal: green -> amber -> red, with an offset per light
  spat_message   the broadcast, modelled on the SAE J2735 SPaT (signal phase and timing) and MAP
                 messages: for every light its id, where its stop line is (track position s), the
                 current state, the time until it changes and the phase durations

Vehicle side:
  V2IPlanner     for every light ahead (within lookahead) decides STOP or GO and turns that into a
                 speed limit:
    GREEN    no limit
    AMBER    1. stopping would need more than b_max            -> go (dilemma if we cannot clear
                                                                   the line before red either)
             2. we clear the line before red AND stopping would
                need more than the comfortable b_comf          -> go (no needless hard stop)
             3. otherwise                                      -> stop
             (a decision made on amber is kept for that approach: no changing our mind)
    RED      stop (unless we already committed to go on amber)
    UNKNOWN  no message for longer than timeout: fail-safe, treat the light as red
    stopping speed limit  v = sqrt(2 * b * (distance to the line - stop_margin)), with
             b = b_comf if that is enough, otherwise the braking actually needed (<= b_max)

  GLOSA (green light optimal speed advisory), when glosa=True: the light also says WHEN it will
  change. If at cruise speed we would arrive on red, on amber (later than its first second) or
  less than arrive_margin after it turns green, find the next green and slow down early so we
  arrive arrive_margin seconds after it turns green:
        advised speed = distance / (time until that green + arrive_margin)
  as long as that is at least v_min (otherwise we would crawl: stop normally instead). Arriving
  on green at speed instead of stopping saves time and the energy thrown away by braking.
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


def phase_after(state, ttc, durations, tau):
    """State of a light tau seconds from now, given its state now, time to change and phase lengths."""
    order = ['GREEN', 'AMBER', 'RED']
    i, left = order.index(state), ttc
    while tau >= left:
        tau -= left
        i = (i + 1) % 3
        left = durations[order[i]]
    return order[i]


def next_green_start(state, ttc, durations):
    """Seconds from now until the light next turns green (0 if it is green now)."""
    if state == 'GREEN':
        return 0.0
    if state == 'AMBER':
        return ttc + durations['RED']
    return ttc
class V2IPlanner:
    def __init__(self, b_comf=0.5, b_max=1.0, stop_margin=0.10, lookahead=15.0, timeout=1.0,
                 glosa=True, cruise=0.5, v_min=0.15, arrive_margin=1.0):
        self.b_comf, self.b_max, self.margin = b_comf, b_max, stop_margin
        self.lookahead, self.timeout = lookahead, timeout
        self.glosa, self.cruise, self.v_min, self.arrive_margin = glosa, cruise, v_min, arrive_margin
        self.map = {}          # id -> stop_s (learned from the messages, kept if they stop)
        self.last = {}         # id -> (state, time_to_change, receive time)
        self.durations = {}    # id -> {'GREEN': s, 'AMBER': s, 'RED': s}
        self.decision = {}     # id -> (STOP | GO, braking used) for the current approach
        self.advising = {}     # id -> advised speed while GLOSA is slowing us down
        self.events = []       # human-readable decisions since the last call

    def receive(self, lights, t):
        for l in lights:
            self.map[l['id']] = float(l['stop_s'])
            self.last[l['id']] = (l['state'], float(l.get('time_to_change', 0.0)), t)
            if 'green' in l:
                self.durations[l['id']] = {'GREEN': float(l['green']), 'AMBER': float(l['amber']),
                                           'RED': float(l['red'])}

    def state_of(self, lid, t):
        if lid not in self.last or t - self.last[lid][2] > self.timeout:
            return 'UNKNOWN', math.inf
        state, ttc, t_rx = self.last[lid]
        return state, max(0.0, ttc - (t - t_rx))

    def glosa_speed(self, lid, state, ttc, dist):
        """Advised speed to arrive on green, or None (no advice needed / not possible)."""
        dur = self.durations.get(lid)
        if not self.glosa or dur is None or state == 'UNKNOWN' or dist < 0.3:
            return None
        arrive = dist / self.cruise                       # when we would get there at cruise speed
        early = phase_after(state, ttc, dur, max(0.0, arrive - self.arrive_margin))
        at = phase_after(state, ttc, dur, arrive)
        if at == 'GREEN' and early == 'GREEN':
            return None                                   # green, with a margin: no advice needed
        if at == 'AMBER' and phase_after(state, ttc, dur, max(0.0, arrive - 1.0)) == 'GREEN':
            return None                                   # first second of amber: we clear it anyway
        if state == 'GREEN':                              # it will change before we get there
            t_green = ttc + dur['AMBER'] + dur['RED']
        else:
            t_green = next_green_start(state, ttc, dur)
        v = dist / (t_green + self.arrive_margin)
        return v if self.v_min <= v < self.cruise else None

    def update(self, s_front, v, t):
        """Returns (speed limit or -1, id of the light that limits us or None)."""
        v = max(0.0, v)
        limit, who = math.inf, None
        for lid, stop_s in self.map.items():
            dist = ahead(stop_s, s_front)
            if dist < -0.05 or dist > self.lookahead:
                self.decision.pop(lid, None)        # passed it, or still far away: new approach later
                self.advising.pop(lid, None)
                continue
            state, ttc = self.state_of(lid, t)
            # GLOSA: slow down early so the light is green when we get there
            adv = self.glosa_speed(lid, state, ttc, dist)
            if adv is not None and self.decision.get(lid, ('', 0))[0] != 'GO':
                if lid not in self.advising:
                    self.events.append(f'GLOSA light {lid}: {state} now, would arrive on red/amber -> slow to '
                                       f'{adv:.2f} m/s to arrive just after it turns green ({dist:.2f} m ahead)')
                self.advising[lid] = adv
                if adv < limit:
                    limit, who = adv, lid
            elif lid in self.advising:
                self.events.append(f'GLOSA light {lid}: advice ended ({state}, {dist:.2f} m ahead)')
                self.advising.pop(lid)
            room = max(dist - self.margin, 0.0)
            need = v * v / (2 * room) if room > 1e-3 else math.inf
            old = self.decision.get(lid)
            if state == 'GREEN':
                if old is not None and old[0] == 'STOP':
                    self.events.append(f'light {lid} GREEN {dist:.2f} m ahead -> go')
                self.decision.pop(lid, None)
                continue
            if old is None:
                clears = v > 0.05 and dist / v + 0.3 <= ttc        # past the line before red
                if state == 'AMBER' and need > self.b_max:
                    old = ('GO', 0.0)
                    self.events.append(f'light {lid} AMBER {dist:.2f} m ahead at {v:.2f} m/s: stopping would need '
                                       f'{need:.2f} m/s2 > {self.b_max} -> go'
                                       + ('' if clears else ' (DILEMMA: cannot clear the line before red either)'))
                elif state == 'AMBER' and clears and need > self.b_comf:
                    old = ('GO', 0.0)
                    self.events.append(f'light {lid} AMBER {dist:.2f} m ahead at {v:.2f} m/s: clears the line '
                                       f'{ttc - dist / v:.1f} s before red, stopping would need {need:.2f} m/s2 -> go')
                elif dist <= 6.0 or state == 'UNKNOWN':
                    b = min(self.b_max, max(self.b_comf, need))
                    old = ('STOP', b)
                    why = 'no V2I message: fail-safe' if state == 'UNKNOWN' else state
                    self.events.append(f'light {lid} {why} {dist:.2f} m ahead at {v:.2f} m/s -> stop '
                                       f'(braking {max(need, 0.0) if math.isfinite(need) else 0.0:.2f} m/s2)')
                else:
                    continue                                  # far away: decide when within 6 m
                self.decision[lid] = old
            if old[0] == 'STOP':
                lim = math.sqrt(2 * old[1] * room)
                if lim < limit:
                    limit, who = lim, lid
        return (limit if math.isfinite(limit) else -1.0), who
