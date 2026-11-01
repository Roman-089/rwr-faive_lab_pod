# 06_calm: LOWFRIC 512-env checks (-upright-v0, hand 0.3-0.6 / cube 0.3-0.5, 512 envs x 1200 steps, seed 42, cmd 1.5)

| checkpoint | why | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| model_5800 | candidate | 13.52 | 88.3 | 0.143 | 11.94 | 8.6 | 0.152 | 98.8 | 97.9 |
| model_6000 | candidate | 13.43 | 86.5 | 0.152 | 11.62 | 8.2 | 0.170 | 98.9 | 97.7 |
| 06_calm2 model_6200 | candidate | 13.38 | 87.3 | 0.139 | 11.68 | 8.1 | 0.148 | 98.9 | 97.5 |

## 06_calm3: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-400.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference: A_07_wp_rf_ft model_4600 | 11.41 | 90.0 | 0.109 | 10.27 | 6.1 | 0.079 | 93.2 | 96.5 | - | - | - |
| reference: 07_wp_rf_ft_seed1 model_4900 | 11.31 | 90.4 | 0.104 | 10.23 | 5.8 | 0.075 | 94.4 | 96.9 | - | - | - |
| reference: DTU 07region_s1_m5500 | not on the pod, run on DTU | | | | | | | | | | |
| 06_calm3 model_6400 | 12.97 | 84.8 | 0.180 | 10.99 | 7.1 | 0.138 | 98.8 | 96.5 | 15.84 | 84.6 | 96.3 |

## 06_calm3_w80: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-400.0 env.rewards.wrist_vel.weight=-16.0 env.rewards.wrist_vel_l1.weight=-80.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3 model_6500 | 12.58 | 88.7 | 0.129 | 11.16 | 6.7 | 0.122 | 99.0 | 98.8 | 15.94 | 83.8 | 96.9 |
| 06_calm3 model_6600 | 12.53 | 86.7 | 0.148 | 10.86 | 6.3 | 0.116 | 98.9 | 96.9 | 15.79 | 84.4 | 96.9 |
| 06_calm3 model_6700 | 12.83 | 89.1 | 0.133 | 11.42 | 6.6 | 0.114 | 99.0 | 97.5 | 15.95 | 86.1 | 97.5 |
| 06_calm3_w80 model_6300 | 13.13 | 87.7 | 0.148 | 11.52 | 8.0 | 0.137 | 98.9 | 97.5 | 16.00 | 83.8 | 96.3 |
| 06_calm3 model_6800 | 12.76 | 88.5 | 0.129 | 11.29 | 6.5 | 0.115 | 98.9 | 98.0 | 15.84 | 86.1 | 98.6 |
| 06_calm3_w80 model_6400 | 13.13 | 89.5 | 0.125 | 11.74 | 9.0 | 0.124 | 98.8 | 97.7 | 16.00 | 84.8 | 96.5 |
| 06_calm3 model_6900 | 12.87 | 89.8 | 0.113 | 11.56 | 6.9 | 0.118 | 99.0 | 99.2 | 16.06 | 85.7 | 98.2 |
| 06_calm3_w80 model_6500 | 13.01 | 87.1 | 0.146 | 11.34 | 9.0 | 0.112 | 98.8 | 98.2 | 15.99 | 86.7 | 96.9 |
| 06_calm3 model_7000 | 12.93 | 90.0 | 0.109 | 11.64 | 7.2 | 0.113 | 98.8 | 98.2 | 16.02 | 86.9 | 99.0 |
| 06_calm3 model_7100 | 12.93 | 90.0 | 0.107 | 11.64 | 7.9 | 0.125 | 98.7 | 98.2 | 16.07 | 86.5 | 98.0 |

## 06_calm3_t800: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_w80 model_6600 | 13.15 | 88.9 | 0.135 | 11.69 | 9.8 | 0.111 | 98.8 | 98.2 | 16.09 | 84.4 | 97.1 |
| 06_calm3 model_7199 | 13.18 | 90.4 | 0.105 | 11.92 | 8.7 | 0.130 | 98.7 | 97.7 | 16.25 | 85.9 | 98.6 |
| 06_calm3_w80 model_6700 | 13.49 | 88.7 | 0.141 | 11.96 | 10.4 | 0.115 | 98.6 | 97.7 | 16.31 | 85.5 | 97.7 |
| 06_calm3_t800 model_5900 | 12.60 | 86.9 | 0.148 | 10.95 | 6.3 | 0.121 | 99.0 | 98.2 | 15.85 | 84.4 | 97.1 |
| 06_calm3_w80 model_6800 | 13.43 | 89.6 | 0.117 | 12.04 | 10.2 | 0.106 | 98.7 | 98.8 | 16.49 | 85.2 | 97.9 |
| 06_calm3_t800 model_6000 | 12.30 | 88.9 | 0.119 | 10.93 | 5.6 | 0.108 | 98.9 | 96.7 | 15.47 | 85.4 | 96.9 |
| 06_calm3_w80 model_6900 | 13.46 | 91.4 | 0.098 | 12.30 | 10.9 | 0.109 | 98.8 | 99.2 | 16.40 | 85.2 | 98.6 |
| 06_calm3_t800 model_6100 | 12.25 | 85.4 | 0.172 | 10.45 | 5.2 | 0.099 | 98.9 | 96.3 | 15.30 | 86.3 | 97.9 |
| 06_calm3_w80 model_7000 | 13.57 | 87.9 | 0.141 | 11.93 | 12.0 | 0.102 | 98.5 | 98.4 | 16.60 | 84.4 | 98.0 |

## tilt400: all-env 512-env LOWFRIC (07 line, kl 0.005, cube_tilt -400); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| reference: A_07_wp_rf_ft model_4600 | 11.41 | 90.0 | 0.109 | 10.27 | 6.1 | 0.079 | 93.2 | 96.5 |
| reference: 07_wp_rf_ft_seed1 model_4900 | 11.31 | 90.4 | 0.104 | 10.23 | 5.8 | 0.075 | 94.4 | 96.9 |
| tilt400 model_4600 (candidate) | 11.52 | 90.2 | 0.100 | 10.40 | 6.1 | 0.075 | 92.2 | 96.7 |
