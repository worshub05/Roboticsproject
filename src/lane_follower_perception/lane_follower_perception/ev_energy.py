"""
EV energy model: how much battery energy the robot uses, treating it as a small electric car.

Forces at the wheels (longitudinal, along the road):
    F = m * a                     accelerating (negative when slowing down)
      + C_rr * m * g * cos(th)    rolling resistance (tyres)
      + 0.5 * rho * CdA * v^2     air drag (tiny at 0.5 m/s, kept for completeness)
      + m * g * sin(th)           road slope (0 on our flat track)
    P_wheel = F * v

Battery power:
    P_wheel > 0   motor drives:   P_batt = P_wheel / eta_drive          (motor, ESC and gearbox losses)
    P_wheel < 0   car slows down: part of it is recovered (regenerative braking),
                  P_batt = - eta_regen * min(-P_wheel, p_regen_max)     only above v_regen_min;
                  everything else is lost as heat in the friction brakes
    plus P_aux    computer and sensors, all the time while the car is on

Energies are reported in joules and as J per metre driven. Traction energy (without P_aux)
is what the driving style changes; P_aux only depends on how long the drive takes.

Pure python (no ROS), so it can be tested offline and used by the energy node in Gazebo.
"""
import math
from dataclasses import dataclass

G = 9.81


@dataclass
class EVParams:
    mass: float = 3.0           # kg, 1:10 car with battery and computer
    c_rr: float = 0.02          # rolling resistance coefficient (rubber tyres on a hard floor)
    cda: float = 0.018          # m2, drag coefficient * frontal area (0.6 * 0.2 m * 0.15 m)
    rho: float = 1.2            # kg/m3, air density
    eta_drive: float = 0.6      # battery -> wheel efficiency when driving (small brushed motor + gearbox)
    eta_regen: float = 0.5      # wheel -> battery efficiency when braking with the motor
    p_regen_max: float = 10.0   # W, most power the motor can push back into the battery
    v_regen_min: float = 0.1    # m/s, below this the motor cannot regenerate: friction brakes only
    regen: bool = True          # regenerative braking on / off
    p_aux: float = 0.0          # W, computer and sensors (e.g. 10 W for a Jetson); 0 = traction only


def wheel_force(v, a, p, slope=0.0):
    """Force the wheels must give (N) at speed v (m/s) and acceleration a (m/s2)."""
    v = max(0.0, v)
    rolling = p.c_rr * p.mass * G * math.cos(slope) if v > 1e-3 else 0.0
    return p.mass * a + rolling + 0.5 * p.rho * p.cda * v * v + p.mass * G * math.sin(slope)


def battery_power(v, a, p, slope=0.0):
    """(battery power W (+ = draws, - = charges), wheel power W, power lost in friction brakes W)."""
    p_wheel = wheel_force(v, a, p, slope) * max(0.0, v)
    if p_wheel >= 0.0:
        return p_wheel / p.eta_drive + p.p_aux, p_wheel, 0.0
    braking = -p_wheel
    recovered = min(braking, p.p_regen_max) if (p.regen and v >= p.v_regen_min) else 0.0
    return -p.eta_regen * recovered + p.p_aux, p_wheel, braking - recovered


class EnergyMeter:
    """
    Integrates the energy from a stream of speed samples (odometry), step by step.
    The acceleration is taken from a smoothed speed, so odometry noise does not count as
    accelerating and braking (same idea as the braking-energy referee at the traffic lights).
    """
    def __init__(self, params=None, alpha=0.3):
        self.p = params or EVParams()
        self.alpha = alpha
        self.vf, self.t_prev = None, None
        self.reset()

    def reset(self):
        self.e_batt = 0.0       # J, net from the battery (drive + aux - recovered)
        self.e_drive = 0.0      # J, battery energy into driving
        self.e_regen = 0.0      # J, energy put back into the battery
        self.e_friction = 0.0   # J, lost in the friction brakes
        self.e_aux = 0.0        # J, computer and sensors
        self.e_wheel_pos = 0.0  # J, positive work at the wheels
        self.distance = 0.0     # m
        self.time = 0.0         # s

    def step(self, v, t, slope=0.0):
        """v: speed (m/s) at time t (s). Returns the battery power (W) of this step."""
        v = max(0.0, v)
        if self.t_prev is None or t <= self.t_prev:
            self.vf, self.t_prev = v, t
            return 0.0
        dt = min(0.5, t - self.t_prev)
        vf = v if self.alpha >= 1.0 else self.vf + self.alpha * (v - self.vf)
        a = (vf - self.vf) / dt
        v_mid = 0.5 * (vf + self.vf)
        p_batt, p_wheel, p_fric = battery_power(v_mid, a, self.p, slope)
        self.e_batt += p_batt * dt
        self.e_aux += self.p.p_aux * dt
        traction = p_batt - self.p.p_aux
        if traction >= 0.0:
            self.e_drive += traction * dt
        else:
            self.e_regen -= traction * dt
        self.e_friction += p_fric * dt
        self.e_wheel_pos += max(0.0, p_wheel) * dt
        self.distance += v_mid * dt
        self.time += dt
        self.vf, self.t_prev = vf, t
        return p_batt

    def traction(self):
        """Net battery energy for driving (J): drive minus recovered, without computer and sensors."""
        return self.e_drive - self.e_regen

    def summary(self):
        d = max(self.distance, 1e-9)
        return dict(distance_m=self.distance, time_s=self.time, traction_J=self.traction(),
                    traction_J_per_m=self.traction() / d, total_J=self.e_batt, total_J_per_m=self.e_batt / d,
                    drive_J=self.e_drive, regen_J=self.e_regen, friction_J=self.e_friction, aux_J=self.e_aux,
                    regen_share=self.e_regen / self.e_drive if self.e_drive > 0 else 0.0)

    def line(self):
        s = self.summary()
        return (f'distance {s["distance_m"]:.1f} m in {s["time_s"]:.0f} s, traction {s["traction_J"]:.1f} J '
                f'({s["traction_J_per_m"]:.2f} J/m), recovered by regen {s["regen_J"]:.1f} J '
                f'({100 * s["regen_share"]:.0f}% of the drive energy), lost in friction brakes {s["friction_J"]:.1f} J, '
                f'total incl. computer {s["total_J"]:.1f} J ({s["total_J_per_m"]:.2f} J/m)')


def trace_energy(speeds, dt, params=None, alpha=1.0):
    """Energy of a whole speed trace sampled every dt seconds. Returns EnergyMeter.summary()."""
    m = EnergyMeter(params, alpha=alpha)
    for k, v in enumerate(speeds):
        m.step(v, k * dt)
    return m.summary()
