# LOWFRIC 512-env re-eval (2026-10-31T17:43:13Z): -upright-v0, hand 0.3-0.6 / cube 0.3-0.5, 512 envs x 1200 steps, seed 42

Last column: drop-free at t >= 2 s (envs without a drop / reset after the spawn catch; cube_motion --steady), the number that matters on the hand, where the cube is placed by hand.

| candidate | checkpoint | rpm | drop-free % | drops/min | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|
| A | 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 | 11.41 | 90.0 | 0.109 | 0.06 | 93.2 | 96.5 |
| B | 2026-10-31_13-36-20_07_wp_rf2_ft_r6k/model_5200 | 11.15 | 89.3 | 0.129 | 0.07 | 91.5 | 94.7 |
| 07ft_m5200 | 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_5200 | 11.16 | 91.4 | 0.100 | 0.06 | 89.7 | 95.5 |
| 07ft_s1_m4900 | 2026-10-31_13-13-29_07_wp_rf_ft_seed1_r6k/model_4900 | 11.39 | 92.2 | 0.082 | 0.07 | 94.2 | 98.0 |
| region200c_m4800 | 2026-10-31_16-46-45_07ft_region200c_ft_r6k/model_4800 | 11.48 | 90.0 | 0.111 | 0.07 | 94.7 | 96.9 |
| region200_m5000 | 2026-10-31_16-17-44_07ft_region200_ft_r6k/model_5000 | 11.40 | 90.6 | 0.105 | 0.07 | 95.2 | 96.3 |
| roll10_m4600 | 2026-10-31_15-44-04_07_wp_rf_roll10_ft_r6k/model_4600 | 11.48 | 91.6 | 0.102 | 0.07 | 93.3 | 96.9 |
| roll10_m5000 | 2026-10-31_15-44-04_07_wp_rf_roll10_ft_r6k/model_5000 | 11.80 | 90.0 | 0.113 | 0.08 | 90.6 | 95.9 |
