# Final experiments (Gazebo, 18 Nov 2026)

Unattended session (`tools/run_experiments.sh`): Gazebo restarted before every run, fixed run
length, summary and all node logs saved per run. 19 measurement runs (E01–E19) without a camera,
5 video takes (V1–V5) with a chase camera. E03, E04 and V2 were re-run with a longer time
(300 s / 200 s), because about 45 s of each run goes on starting Gazebo and the nodes and the robot
had not yet reached the slow car. **0 collisions in all runs.**

Energy: live EV energy model on odometry (3 kg, C_rr 0.02, battery to wheel 60 %, regen 50 %),
traction energy per metre, without the computer.

## Overtaking

| Run | Controller | Overtakes | Collisions | Smallest RSS margin | Energy |
|---|---|---|---|---|---|
| E01 | MPC | 1 | 0 | 0.27 m | 1.04 J/m |
| E02 | tracker | 1 | 0 | 0.08 m | 1.03 J/m |

## RSS with a broken ACC (follows a slow car with almost no gap)

| Run | RSS | Dangerous time | Closest approach | AEB full brakes | Energy |
|---|---|---|---|---|---|
| E03 | off | **51 s (23.5 %)**, 63 situations | **0.15 m** | 64 | 1.12 J/m |
| E04 | on | **0 s** | **0.25 m** | 2 | 0.99 J/m |

RSS removes every dangerous situation, and the robot uses 12 % less energy because it no longer
brakes hard again and again (2.8 J recovered + 0.6 J lost in the brakes without RSS).

## AEB: a car cuts in and brakes hard / a stopped car appears

| Run | Gap | Safety | Closest approach (cut-in / pop-up) | Collisions |
|---|---|---|---|---|
| E05 | 0.40 m | none | 0.08 / 0.08 m | 0 |
| E06 | 0.40 m | RSS | 0.18 / 0.18 m | 0 |
| E07 | 0.40 m | RSS + AEB | 0.18 / **0.23 m** | 0 |
| E08 | 0.25 m | RSS | 0.18 / 0.10 m | 0 |
| E09 | 0.25 m | RSS + AEB | 0.20 / **0.13 m** | 0 |

## V2I traffic lights (two lights per lap)

| Run | Setup | Red crossings | Stops | Speed at the line | Energy |
|---|---|---|---|---|---|
| E10 | no V2I | **1** (+1 amber) of 3 | 0 | 0.50 m/s | 0.99 J/m |
| E11 | V2I stop | 0 | 2 (7.3–8.2 cm before the line) | 0.34 m/s | 1.00 J/m |
| E12 | V2I + GLOSA | 0 | **0** | 0.50 m/s | 0.99 J/m |
| E13 | GLOSA, 29 % lost + 0.3 s delay | 0 | 0 | 0.50 m/s | 0.99 J/m |
| E14 | V2I blackout | 0 | fail-safe stop 8.1 cm before the line, stays stopped | – | 0.99 J/m |

At 0.5 m/s the V2I stops are gentle (0.02–0.03 m/s², slower than rolling resistance alone), so
they cost almost no extra energy on this 1:10 car. On a real car one stop at 50 km/h costs about
1 km of driving (see `ev_energy_test.py`, section 5).

## Platoon (leader, simulated car 1, our robot; three braking events)

| Run | Mode | Amplification | Smallest gap (robot) | Energy |
|---|---|---|---|---|
| E15 | ACC, 1.0 s time gap | 0.89 (stable) | 0.74 m | 1.07 J/m |
| E16 | ACC, 0.3 s time gap | **1.13 (unstable)** | 0.37 m | 1.06 J/m |
| E17 | CACC, 0.3 s time gap | **0.94 (stable)** | 0.49 m | **1.02 J/m** |
| E18 | CACC, 50 % lost + 0.2 s delay | 1.01 (stable, gap grows to 0.88 m) | 0.56 m | 1.01 J/m |
| E19 | CACC, V2V blackout | 0.94 (falls back to ACC) | 0.50 m | 1.05 J/m |

CACC keeps the short gap stable and uses about 5 % less energy than ACC, as the offline model
predicted (`ev_energy_test.py`, section 4).

## Video takes

| Video | GIF | Segment |
|---|---|---|
| V1 | `gifs/1_overtaking_mpc.gif` | 26–50 s, 2× speed |
| V2 | `gifs/2_rss_broken_acc.gif` | 64–100 s, 3× speed |
| V3 | `gifs/3_aeb_cut_in.gif` | 16–30 s, real time |
| V4 | `gifs/4_v2i_glosa_lights.gif` | 18–40 s, 2× speed |
| V5 | `gifs/5_cacc_platoon.gif` | 6–36 s, 2× speed |

Video takes are not measurements: the chase camera adds load (V5 shows one braking event with
amplification 1.50; the measured value is E17).

## Notes

- The traffic-light referee's "braking energy" (mJ/kg) is inflated by speed noise in these runs
  (e.g. 202 mJ/kg per light in E12, more than a full stop from 0.5 m/s releases). The energy
  node smooths odometry and passed the noise check, so its numbers are the ones used.
- Logs of every run: `results/experiments/<run>/`, summaries in `summary_main_session.txt`
  (E01–E19, V1–V5) and `summary.txt` (re-runs E03, E04, V2).
