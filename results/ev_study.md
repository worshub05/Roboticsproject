# EV integration study

## 1. Energy of every Gazebo run (traction = without the computer)

| Run | Scenario | Distance | Time | Traction | Recovered | Friction brakes | With a 10 W computer |
|---|---|---|---|---|---|---|---|
| E01 | Overtaking with MPC steering | 37.1 m | 95 s | 1.04 J/m | 0.6 J | 0.0 J | 26.6 J/m |
| E02 | Overtaking with the tracker controller | 37.7 m | 95 s | 1.02 J/m | 0.4 J | 0.0 J | 26.2 J/m |
| E03 | Broken ACC, RSS off | 55.3 m | 214 s | 1.12 J/m | 2.8 J | 0.6 J | 39.8 J/m |
| E04 | Broken ACC, RSS on | 55.0 m | 213 s | 0.99 J/m | 0.2 J | 0.0 J | 39.7 J/m |
| E05 | Cut-in 0.40 m: no RSS, no AEB | 21.5 m | 55 s | 1.05 J/m | 0.3 J | 0.0 J | 26.6 J/m |
| E06 | Cut-in 0.40 m: RSS only | 20.4 m | 61 s | 1.05 J/m | 0.3 J | 0.0 J | 31.0 J/m |
| E07 | Cut-in 0.40 m: RSS + AEB | 27.2 m | 69 s | 1.04 J/m | 0.3 J | 0.0 J | 26.4 J/m |
| E08 | Cut-in 0.25 m: RSS only | 26.9 m | 69 s | 1.04 J/m | 0.4 J | 0.0 J | 26.7 J/m |
| E09 | Cut-in 0.25 m: RSS + AEB | 25.9 m | 69 s | 1.04 J/m | 0.3 J | 0.0 J | 27.7 J/m |
| E10 | Traffic lights without V2I | 105.9 m | 217 s | 0.99 J/m | 0.0 J | 0.0 J | 21.5 J/m |
| E11 | Traffic lights: V2I stop only | 103.1 m | 216 s | 1.00 J/m | 0.3 J | 0.0 J | 21.9 J/m |
| E12 | Traffic lights: V2I + GLOSA | 103.8 m | 217 s | 0.99 J/m | 0.1 J | 0.0 J | 21.9 J/m |
| E13 | V2I + GLOSA, 30% lost + 0.3 s delay | 105.8 m | 221 s | 0.99 J/m | 0.1 J | 0.0 J | 21.9 J/m |
| E14 | V2I blackout (fail-safe) | 47.5 m | 210 s | 0.99 J/m | 0.1 J | 0.0 J | 45.2 J/m |
| E15 | Platoon: ACC, 1.0 s time gap | 30.4 m | 82 s | 1.07 J/m | 0.9 J | 0.0 J | 28.0 J/m |
| E16 | Platoon: ACC, 0.3 s time gap | 32.4 m | 84 s | 1.06 J/m | 0.8 J | 0.0 J | 27.0 J/m |
| E17 | Platoon: CACC, 0.3 s time gap | 33.4 m | 88 s | 1.02 J/m | 0.3 J | 0.0 J | 27.4 J/m |
| E18 | CACC, 50% lost + 0.2 s delay | 32.3 m | 84 s | 1.01 J/m | 0.1 J | 0.0 J | 27.0 J/m |
| E19 | CACC, V2V blackout | 33.0 m | 88 s | 1.05 J/m | 0.7 J | 0.0 J | 27.7 J/m |
| V1 | Overtaking with MPC steering | 16.2 m | 53 s | 1.12 J/m | 0.7 J | 0.0 J | 33.8 J/m |
| V2 | RSS keeps a safe distance (broken ACC) | 44.1 m | 139 s | 1.01 J/m | 0.4 J | 0.1 J | 32.5 J/m |
| V3 | Emergency braking for a cut-in car | 12.0 m | 41 s | 1.12 J/m | 0.3 J | 0.0 J | 35.3 J/m |
| V4 | Traffic lights with V2I + GLOSA | 32.8 m | 74 s | 1.03 J/m | 0.3 J | 0.0 J | 23.6 J/m |
| V5 | CACC platoon, 0.3 s time gap | 17.8 m | 48 s | 1.06 J/m | 0.3 J | 0.0 J | 28.0 J/m |

## 2. With and without each feature

| Comparison | Setup | Traction | vs first | With a 10 W computer | vs first |
|---|---|---|---|---|---|
| Safety with a broken ACC | E03 RSS off | 1.121 J/m | +0.0 % | 39.8 J/m | +0.0 % |
|  | E04 RSS on | 0.995 J/m | -11.3 % | 39.7 J/m | -0.2 % |
| Cut-in 0.40 m | E05 no RSS, no AEB | 1.051 J/m | +0.0 % | 26.6 J/m | +0.0 % |
|  | E06 RSS | 1.049 J/m | -0.2 % | 31.0 J/m | +16.2 % |
|  | E07 RSS + AEB | 1.040 J/m | -1.0 % | 26.4 J/m | -0.8 % |
| Traffic lights | E10 no V2I (runs red) | 0.992 J/m | +0.0 % | 21.5 J/m | +0.0 % |
|  | E11 V2I stop | 0.998 J/m | +0.7 % | 21.9 J/m | +2.2 % |
|  | E12 V2I + GLOSA | 0.993 J/m | +0.2 % | 21.9 J/m | +1.9 % |
|  | E13 GLOSA, 29 % lost | 0.993 J/m | +0.2 % | 21.9 J/m | +1.9 % |
| Platoon | E15 ACC 1.0 s | 1.072 J/m | +0.0 % | 28.0 J/m | +0.0 % |
|  | E16 ACC 0.3 s | 1.059 J/m | -1.3 % | 27.0 J/m | -3.8 % |
|  | E17 CACC 0.3 s | 1.021 J/m | -4.8 % | 27.4 J/m | -2.4 % |
|  | E18 CACC, bad radio | 1.012 J/m | -5.6 % | 27.0 J/m | -3.7 % |
|  | E19 CACC, blackout | 1.048 J/m | -2.2 % | 27.7 J/m | -1.2 % |
| Overtaking | E01 MPC | 1.038 J/m | +0.0 % | 26.6 J/m | +0.0 % |
|  | E02 tracker | 1.024 J/m | -1.3 % | 26.2 J/m | -1.6 % |

## 3. Regenerative braking (offline model, 60 s, a stop every 10 s)

| Regen efficiency | Gentle stop-and-go | Hard stop-and-go |
|---|---|---|
| 0% | 1.050 J/m | 1.151 J/m |
| 50% | 1.031 J/m | 1.105 J/m |
| 70% | 1.024 J/m | 1.086 J/m |

## 4. The same events at full scale (cost in metres of steady cruising)

| Vehicle | Speed | Cruising | One hard stop | One gentle (coasting) stop | 10 s wait (computer / auxiliaries) | Computer share at cruise |
|---|---|---|---|---|---|---|
| our 1:10 car | 0.5 m/s | 0.99 J/m | 0.5 m | 0.0 m | 101 m | 95 % |
| electric car, 50 km/h | 13.9 m/s | 289.44 J/m | 393.8 m | 0.0 m | 52 m | 27 % |

Assumptions: 1:10 car as in ev_energy.py with a 10 W computer; electric car 1800 kg, C_rr 0.01,
CdA 0.6 m2, battery to wheel 85 %, regen 65 % (80 % of a hard stop braked by the motor), 1.5 kW
auxiliaries and self-driving computer. A coasting stop lets rolling resistance use up the speed,
so it costs almost nothing extra in traction; only the time spent waiting costs energy.

## 5. Findings

- On the 1:10 car the computer and sensors use about 95 % of the energy at cruise, so energy per
  metre is set mostly by how fast the robot gets there, not by how it brakes.
- Traction energy still shows the effect of the driving style: RSS with a broken ACC uses 11 % less
  (no repeated hard braking), CACC uses about 5 % less than ACC in the platoon.
- V2I stops on this car are gentle enough to coast on rolling resistance, so GLOSA saves no
  measurable traction energy here; its benefit is no stops and no waiting.
- On a real electric car one hard stop from 50 km/h costs about as much as 400 m of driving, so
  every stop that GLOSA avoids, and every braking wave that CACC smooths out, matters far more.
