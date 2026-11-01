# LOWFRIC 512-env re-eval 2 (2026-11-01T01:28:48Z): -upright-v0, hand 0.3-0.6 / cube 0.3-0.5, 512 envs x 1200 steps, seed 42, cmd 1.5 unless noted

Last column: drop-free at t >= 2 s (envs without a drop / reset after the spawn catch; cube_motion --steady), the number that matters on the hand, where the cube is placed by hand.

In region = fraction of alive steps with the cube inside the original circle (-11.6, 13.6) mm, r 27.5 mm. Wrist speed in rad/s (ship rule <= ~0.10-0.14).

| run | checkpoint | cmd | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|
| kl005 | model_4700 | 1.5 | 11.40 | 89.8 | 0.109 | 10.25 | 6.4 | 0.062 | 93.2 | 96.7 |
| kl005 | model_4800 | 1.5 | 11.38 | 89.8 | 0.125 | 10.23 | 6.1 | 0.070 | 92.9 | 95.1 |
| kl005 | model_4900 | 1.5 | 11.55 | 91.6 | 0.100 | 10.58 | 6.1 | 0.072 | 93.2 | 96.1 |
| kl005 | model_5499 | 1.5 | 11.31 | 88.7 | 0.133 | 10.03 | 5.9 | 0.076 | 90.5 | 95.1 |
| region200 | model_5400 | 1.5 | 10.61 | 89.1 | 0.133 | 9.45 | 5.8 | 0.061 | 96.3 | 95.3 |
| region200c | model_5500 | 1.5 | 11.13 | 90.6 | 0.104 | 10.09 | 5.9 | 0.068 | 96.8 | 96.1 |
| region200c | model_5599 | 1.5 | 10.45 | 87.3 | 0.141 | 9.13 | 5.7 | 0.066 | 97.3 | 94.3 |
| region200cs | model_5100 | 1.5 | 11.13 | 92.6 | 0.078 | 10.31 | 5.9 | 0.069 | 96.6 | 96.1 |
| region200cs | model_5599 | 1.5 | 10.58 | 90.0 | 0.117 | 9.53 | 5.9 | 0.070 | 97.0 | 93.8 |
| cmd2 | model_4900 | 1.5 | 11.29 | 91.4 | 0.102 | 10.32 | 5.9 | 0.068 | 94.5 | 96.3 |
| cmd2 | model_4900 | 2.0 | 11.32 | 87.9 | 0.133 | 9.95 | 5.9 | 0.060 | 95.2 | 95.5 |
