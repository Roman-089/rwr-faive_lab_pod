# Drop analysis, 512-env LOWFRIC npz (2026-11-01T01:14:31Z)

Script: ~/rwr_drop_analysis.py on the pod. Circle (-11.6, 13.6) mm, r 27.5 mm; 20 Hz; every reset in this eval is a drop (no time-outs). Wrist angle is only logged for envs 0-3.

## A_m4600  (model_4600_lowfric512.npz; 512 envs x 1200 steps, 56 drops, 24 with a clean 1-s window)

- drops within the first 2 s of an episode: 68 % (an alive step is in its first 2 s 3.5 % of the time)

| quantity | 1 s before drops: mean | p90 | all alive steps: mean | p90 |
|---|---|---|---|---|
| distance from centre, 1-s mean (mm) | 55.2 | 74.3 | 20.7 | 26.3 |
| distance at the last step before the drop (mm) | 187.8 | 315.4 | 20.7 | 26.3 |
| share of the window outside the circle (%) | 57.1 | 100.0 | 5.8 | 0.0 |
| tilt, 1-s mean (deg) | 20.6 | 30.0 | 5.9 | 9.7 |
| tilt, 1-s max (deg) | 45.8 | 50.3 | 5.9 | 9.7 |
| cube |w_z| / |command|, 1-s mean | 2.45 | 6.19 | 0.89 | 1.88 |
| \|wrist speed\| | n/a: no drop in the 4 link-logged envs | | 0.048 | |

Bearing of the cube from the circle centre (last step before the drop vs all alive steps):

| sector | -x | -x-y | -y | +x-y | +x | +x+y | +y | -x+y |
|---|---|---|---|---|---|---|---|---|
| before drops % | 4 | 17 | 21 | 4 | 38 | 8 | 0 | 8 |
| all alive % | 0 | 1 | 91 | 8 | 0 | 0 | 0 | 0 |

Early drops (episode age < 2 s): 38 of 56; 34 in the very first episode (t < 2 s after the eval start), 4 right after a respawn.
Envs with any drop: 51 (drop-free 90.0 %); first drop within 2 s of the eval start: 34 envs, by 0.25-s bin [1, 16, 10, 1, 2, 3, 0, 1]. Envs with a drop after the first 2 s: 18 (drop-free over t >= 2 s: 96.5 %).

Lead-up before the cube leaves the hand (mid-episode drops with a clean window, n=15; departure -> registered drop takes 0.40 s median):

| quantity | 1 s before departure: mean | p90 | all alive steps: mean | p90 |
|---|---|---|---|---|
| distance from centre, 1-s mean (mm) | 18.9 | 27.4 | 20.7 | 26.3 |
| distance at departure - 1 step (mm) | 22.8 | 31.6 | 20.7 | 26.3 |
| share of the window outside the circle (%) | 12.7 | 46.0 | 5.8 | 0.0 |
| tilt, 1-s mean (deg) | 10.2 | 15.9 | 5.9 | 9.7 |
| tilt, 1-s max (deg) | 24.0 | 39.3 | 5.9 | 9.7 |
| cube |w_z| / |command|, 1-s mean | 0.86 | 1.22 | 0.89 | 1.88 |

Exit bearing (cube position at departure) vs all alive steps:

| sector | -x | -x-y | -y | +x-y | +x | +x+y | +y | -x+y |
|---|---|---|---|---|---|---|---|---|
| exits % | 0 | 7 | 40 | 20 | 13 | 13 | 7 | 0 |
| all alive % | 0 | 1 | 91 | 8 | 0 | 0 | 0 | 0 |

Mean profile before departure (s): -2.00: 19 mm / 7 deg, -1.50: 19 mm / 8 deg, -1.00: 20 mm / 13 deg, -0.75: 18 mm / 9 deg, -0.50: 18 mm / 8 deg, -0.25: 19 mm / 10 deg, -0.05: 23 mm / 12 deg

P(drop within 1 s) by distance from the circle centre (mm); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <10 | 10-20 | 20-27.5 | 27.5-35 | >=35 |
|---|---|---|---|---|---|
| alive % | 2.5 | 40.4 | 51.3 | 4.4 | 1.4 |
| per 1000 | 2.43 | 0.25 | 0.20 | 1.03 | 25.36 |

P(drop within 1 s) by tilt (deg); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <5 | 5-10 | 10-15 | 15-25 | >=25 |
|---|---|---|---|---|---|
| alive % | 44.2 | 46.9 | 6.4 | 2.0 | 0.5 |
| per 1000 | 0.26 | 0.23 | 0.84 | 4.94 | 59.44 |

P(drop within 1 s) by |w_z|/|command| (-); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <0.5 | 0.5-0.8 | 0.8-1.2 | 1.2-1.5 | >=1.5 |
|---|---|---|---|---|---|
| alive % | 37.5 | 14.4 | 16.5 | 10.8 | 20.8 |
| per 1000 | 0.63 | 0.41 | 0.47 | 0.42 | 1.22 |

P(drop within 1 s) by cube y offset from the circle centre (mm); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <-20 | -20--10 | -10-0 | 0-10 | >=10 |
|---|---|---|---|---|---|
| alive % | 50.4 | 46.4 | 3.0 | 0.1 | 0.1 |
| per 1000 | 0.50 | 0.19 | 2.92 | 29.95 | 222.22 |

## kl005_m4900  (model_4900_lowfric512.npz; 512 envs x 1200 steps, 51 drops, 26 with a clean 1-s window)

- drops within the first 2 s of an episode: 55 % (an alive step is in its first 2 s 3.5 % of the time)

| quantity | 1 s before drops: mean | p90 | all alive steps: mean | p90 |
|---|---|---|---|---|
| distance from centre, 1-s mean (mm) | 48.4 | 62.3 | 21.0 | 26.1 |
| distance at the last step before the drop (mm) | 129.6 | 171.0 | 21.0 | 26.1 |
| share of the window outside the circle (%) | 60.8 | 100.0 | 5.7 | 0.0 |
| tilt, 1-s mean (deg) | 21.2 | 27.7 | 5.9 | 9.7 |
| tilt, 1-s max (deg) | 44.8 | 50.4 | 5.9 | 9.7 |
| cube |w_z| / |command|, 1-s mean | 2.10 | 3.62 | 0.89 | 1.86 |
| \|wrist speed\| | n/a: no drop in the 4 link-logged envs | | 0.064 | |

Bearing of the cube from the circle centre (last step before the drop vs all alive steps):

| sector | -x | -x-y | -y | +x-y | +x | +x+y | +y | -x+y |
|---|---|---|---|---|---|---|---|---|
| before drops % | 15 | 27 | 4 | 8 | 35 | 8 | 0 | 4 |
| all alive % | 0 | 2 | 91 | 8 | 0 | 0 | 0 | 0 |

Early drops (episode age < 2 s): 28 of 51; 23 in the very first episode (t < 2 s after the eval start), 5 right after a respawn.
Envs with any drop: 43 (drop-free 91.6 %); first drop within 2 s of the eval start: 23 envs, by 0.25-s bin [0, 18, 1, 2, 0, 1, 1, 0]. Envs with a drop after the first 2 s: 20 (drop-free over t >= 2 s: 96.1 %).

Lead-up before the cube leaves the hand (mid-episode drops with a clean window, n=19; departure -> registered drop takes 0.40 s median):

| quantity | 1 s before departure: mean | p90 | all alive steps: mean | p90 |
|---|---|---|---|---|
| distance from centre, 1-s mean (mm) | 20.3 | 25.9 | 21.0 | 26.1 |
| distance at departure - 1 step (mm) | 23.4 | 33.4 | 21.0 | 26.1 |
| share of the window outside the circle (%) | 14.5 | 37.0 | 5.7 | 0.0 |
| tilt, 1-s mean (deg) | 12.5 | 20.9 | 5.9 | 9.7 |
| tilt, 1-s max (deg) | 27.0 | 42.6 | 5.9 | 9.7 |
| cube |w_z| / |command|, 1-s mean | 1.12 | 1.79 | 0.89 | 1.86 |

Exit bearing (cube position at departure) vs all alive steps:

| sector | -x | -x-y | -y | +x-y | +x | +x+y | +y | -x+y |
|---|---|---|---|---|---|---|---|---|
| exits % | 0 | 32 | 26 | 11 | 11 | 11 | 5 | 5 |
| all alive % | 0 | 2 | 91 | 8 | 0 | 0 | 0 | 0 |

Mean profile before departure (s): -2.00: 17 mm / 10 deg, -1.50: 21 mm / 10 deg, -1.00: 18 mm / 11 deg, -0.75: 21 mm / 12 deg, -0.50: 20 mm / 16 deg, -0.25: 20 mm / 10 deg, -0.05: 23 mm / 13 deg

P(drop within 1 s) by distance from the circle centre (mm); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <10 | 10-20 | 20-27.5 | 27.5-35 | >=35 |
|---|---|---|---|---|---|
| alive % | 2.0 | 36.3 | 56.0 | 4.2 | 1.5 |
| per 1000 | 1.16 | 0.45 | 0.26 | 1.48 | 33.14 |

P(drop within 1 s) by tilt (deg); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <5 | 5-10 | 10-15 | 15-25 | >=25 |
|---|---|---|---|---|---|
| alive % | 44.2 | 46.8 | 6.6 | 1.8 | 0.5 |
| per 1000 | 0.29 | 0.24 | 1.01 | 11.17 | 67.82 |

P(drop within 1 s) by |w_z|/|command| (-); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <0.5 | 0.5-0.8 | 0.8-1.2 | 1.2-1.5 | >=1.5 |
|---|---|---|---|---|---|
| alive % | 36.3 | 14.9 | 17.3 | 11.2 | 20.3 |
| per 1000 | 0.73 | 0.67 | 0.45 | 0.56 | 1.86 |

P(drop within 1 s) by cube y offset from the circle centre (mm); share of alive steps, drops per 1000 steps, drop-steps:

| bin | <-20 | -20--10 | -10-0 | 0-10 | >=10 |
|---|---|---|---|---|---|
| alive % | 54.4 | 42.9 | 2.5 | 0.1 | 0.1 |
| per 1000 | 0.78 | 0.42 | 2.22 | 31.01 | 132.08 |

