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
| 06_calm3 model_6500 | 12.58 | 88.7 | 0.129 | 11.16 | 6.7 | 0.122 | 99.0 | 98.8 | 15.94 | 83.8 | 96.9 |
| 06_calm3 model_6600 | 12.53 | 86.7 | 0.148 | 10.86 | 6.3 | 0.116 | 98.9 | 96.9 | 15.79 | 84.4 | 96.9 |
| 06_calm3 model_6700 | 12.83 | 89.1 | 0.133 | 11.42 | 6.6 | 0.114 | 99.0 | 97.5 | 15.95 | 86.1 | 97.5 |
| 06_calm3 model_6800 | 12.76 | 88.5 | 0.129 | 11.29 | 6.5 | 0.115 | 98.9 | 98.0 | 15.84 | 86.1 | 98.6 |
| 06_calm3 model_6900 | 12.87 | 89.8 | 0.113 | 11.56 | 6.9 | 0.118 | 99.0 | 99.2 | 16.06 | 85.7 | 98.2 |
| 06_calm3 model_7000 | 12.93 | 90.0 | 0.109 | 11.64 | 7.2 | 0.113 | 98.8 | 98.2 | 16.02 | 86.9 | 99.0 |
| 06_calm3 model_7100 | 12.93 | 90.0 | 0.107 | 11.64 | 7.9 | 0.125 | 98.7 | 98.2 | 16.07 | 86.5 | 98.0 |
| 06_calm3 model_7199 | 13.18 | 90.4 | 0.105 | 11.92 | 8.7 | 0.130 | 98.7 | 97.7 | 16.25 | 85.9 | 98.6 |

## 06_calm3_w80: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-400.0 env.rewards.wrist_vel.weight=-16.0 env.rewards.wrist_vel_l1.weight=-80.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_w80 model_6300 | 13.13 | 87.7 | 0.148 | 11.52 | 8.0 | 0.137 | 98.9 | 97.5 | 16.00 | 83.8 | 96.3 |
| 06_calm3_w80 model_6400 | 13.13 | 89.5 | 0.125 | 11.74 | 9.0 | 0.124 | 98.8 | 97.7 | 16.00 | 84.8 | 96.5 |
| 06_calm3_w80 model_6500 | 13.01 | 87.1 | 0.146 | 11.34 | 9.0 | 0.112 | 98.8 | 98.2 | 15.99 | 86.7 | 96.9 |
| 06_calm3_w80 model_6600 | 13.15 | 88.9 | 0.135 | 11.69 | 9.8 | 0.111 | 98.8 | 98.2 | 16.09 | 84.4 | 97.1 |
| 06_calm3_w80 model_6700 | 13.49 | 88.7 | 0.141 | 11.96 | 10.4 | 0.115 | 98.6 | 97.7 | 16.31 | 85.5 | 97.7 |
| 06_calm3_w80 model_6800 | 13.43 | 89.6 | 0.117 | 12.04 | 10.2 | 0.106 | 98.7 | 98.8 | 16.49 | 85.2 | 97.9 |
| 06_calm3_w80 model_6900 | 13.46 | 91.4 | 0.098 | 12.30 | 10.9 | 0.109 | 98.8 | 99.2 | 16.40 | 85.2 | 98.6 |
| 06_calm3_w80 model_7000 | 13.57 | 87.9 | 0.141 | 11.93 | 12.0 | 0.102 | 98.5 | 98.4 | 16.60 | 84.4 | 98.0 |
| 06_calm3_w80 model_7100 | 13.65 | 90.8 | 0.104 | 12.39 | 10.6 | 0.105 | 98.7 | 98.8 | 16.58 | 85.2 | 97.5 |
| 06_calm3_w80 model_7199 | 13.63 | 88.3 | 0.127 | 12.03 | 10.6 | 0.112 | 98.7 | 98.6 | 16.51 | 85.7 | 98.2 |

## 06_calm3_t800: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_t800 model_5900 | 12.60 | 86.9 | 0.148 | 10.95 | 6.3 | 0.121 | 99.0 | 98.2 | 15.85 | 84.4 | 97.1 |
| 06_calm3_t800 model_6000 | 12.30 | 88.9 | 0.119 | 10.93 | 5.6 | 0.108 | 98.9 | 96.7 | 15.47 | 85.4 | 96.9 |
| 06_calm3_t800 model_6100 | 12.25 | 85.4 | 0.172 | 10.45 | 5.2 | 0.099 | 98.9 | 96.3 | 15.30 | 86.3 | 97.9 |
| 06_calm3_t800 model_6200 | 12.66 | 86.5 | 0.156 | 10.96 | 5.0 | 0.090 | 98.9 | 98.0 | 15.61 | 85.9 | 97.1 |
| 06_calm3_t800 model_6300 | 12.46 | 89.5 | 0.121 | 11.15 | 4.8 | 0.086 | 99.0 | 98.4 | 15.54 | 85.0 | 97.1 |
| 06_calm3_t800 model_6400 | 12.62 | 88.1 | 0.150 | 11.11 | 4.7 | 0.081 | 99.0 | 98.8 | 15.63 | 85.4 | 98.6 |
| 06_calm3_t800 model_6500 | 12.37 | 87.3 | 0.154 | 10.80 | 4.4 | 0.078 | 99.1 | 97.7 | 15.55 | 84.8 | 97.9 |
| 06_calm3_t800 model_6600 | 12.72 | 87.1 | 0.143 | 11.08 | 4.5 | 0.079 | 99.1 | 98.0 | 15.78 | 84.6 | 98.4 |
| 06_calm3_t800 model_6700 | 12.80 | 88.5 | 0.129 | 11.33 | 4.7 | 0.084 | 99.1 | 98.4 | 15.91 | 85.2 | 97.9 |
| 06_calm3_t800 model_6799 | 12.98 | 87.7 | 0.139 | 11.38 | 4.7 | 0.084 | 99.1 | 98.6 | 15.94 | 85.2 | 98.0 |

## tilt400: all-env 512-env LOWFRIC (07 line, kl 0.005, cube_tilt -400); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| reference: A_07_wp_rf_ft model_4600 | 11.41 | 90.0 | 0.109 | 10.27 | 6.1 | 0.079 | 93.2 | 96.5 |
| reference: 07_wp_rf_ft_seed1 model_4900 | 11.31 | 90.4 | 0.104 | 10.23 | 5.8 | 0.075 | 94.4 | 96.9 |
| tilt400 model_4600 (candidate) | 11.52 | 90.2 | 0.100 | 10.40 | 6.1 | 0.075 | 92.2 | 96.7 |
| tilt400 model_5200 (candidate) | 11.41 | 90.8 | 0.100 | 10.36 | 5.6 | 0.074 | 91.2 | 96.3 |
| tilt400 model_5499 (final) | 11.29 | 90.0 | 0.109 | 10.16 | 5.7 | 0.073 | 90.1 | 96.3 |

## 06_calm3_s1: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-400.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_s1 model_6300 | 13.00 | 87.9 | 0.145 | 11.42 | 7.7 | 0.140 | 99.0 | 98.2 | 16.01 | 85.0 | 96.7 |
| 06_calm3_s1 model_6400 | 12.69 | 87.5 | 0.139 | 11.11 | 7.8 | 0.137 | 99.0 | 97.5 | 15.95 | 83.4 | 97.1 |
| 06_calm3_s1 model_6500 | 12.65 | 87.3 | 0.145 | 11.04 | 7.7 | 0.132 | 98.9 | 97.3 | 16.03 | 86.3 | 98.0 |
| 06_calm3_s1 model_6600 | 12.67 | 87.1 | 0.154 | 11.04 | 8.2 | 0.127 | 98.8 | 96.1 | 16.09 | 87.1 | 97.7 |
| 06_calm3_s1 model_6700 | 13.01 | 87.9 | 0.141 | 11.43 | 8.7 | 0.120 | 99.0 | 97.9 | 16.14 | 85.2 | 97.9 |
| 06_calm3_s1 model_6800 | 12.94 | 89.5 | 0.121 | 11.57 | 8.5 | 0.119 | 99.0 | 98.2 | 16.24 | 84.8 | 97.9 |
| 06_calm3_s1 model_6900 | 13.07 | 88.5 | 0.135 | 11.57 | 8.8 | 0.119 | 98.8 | 97.9 | 16.28 | 85.4 | 98.6 |
| 06_calm3_s1 model_7000 | 12.97 | 89.5 | 0.113 | 11.60 | 8.3 | 0.115 | 99.0 | 99.0 | 16.05 | 85.7 | 97.7 |
| 06_calm3_s1 model_7100 | 13.08 | 89.3 | 0.125 | 11.68 | 8.5 | 0.116 | 98.7 | 98.2 | 16.27 | 85.2 | 98.0 |
| 06_calm3_s1 model_7199 | 13.21 | 86.3 | 0.162 | 11.40 | 9.0 | 0.119 | 98.7 | 97.3 | 16.42 | 85.7 | 98.6 |

## 06_calm3_w80t800: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.rewards.wrist_vel_l1.weight=-80.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_w80t800 model_6300 | 12.61 | 88.9 | 0.121 | 11.21 | 7.0 | 0.127 | 99.1 | 98.6 | 15.69 | 85.5 | 96.5 |
| 06_calm3_w80t800 model_6400 | 12.16 | 88.3 | 0.133 | 10.73 | 6.8 | 0.111 | 99.1 | 97.3 | 15.57 | 87.7 | 98.4 |
| 06_calm3_w80t800 model_6500 | 12.52 | 86.3 | 0.164 | 10.81 | 6.7 | 0.104 | 98.9 | 96.7 | 15.66 | 83.6 | 97.5 |
| 06_calm3_w80t800 model_6600 | 12.12 | 88.1 | 0.127 | 10.68 | 5.9 | 0.088 | 99.1 | 98.6 | 15.73 | 85.4 | 98.4 |
| 06_calm3_w80t800 model_6700 | 12.00 | 88.3 | 0.139 | 10.59 | 5.1 | 0.084 | 99.1 | 97.5 | 15.44 | 85.7 | 97.9 |
| 06_calm3_w80t800 model_6800 | 12.15 | 88.3 | 0.143 | 10.73 | 5.2 | 0.086 | 99.2 | 97.9 | 15.64 | 86.1 | 97.3 |
| 06_calm3_w80t800 model_6900 | 12.50 | 88.3 | 0.135 | 11.03 | 5.3 | 0.089 | 99.1 | 98.4 | 15.64 | 84.6 | 98.0 |
| 06_calm3_w80t800 model_7000 | 12.50 | 89.1 | 0.117 | 11.13 | 5.3 | 0.088 | 99.1 | 98.4 | 15.80 | 87.1 | 97.9 |
| 06_calm3_w80t800 model_7100 | 12.89 | 89.6 | 0.113 | 11.56 | 5.9 | 0.089 | 99.2 | 99.0 | 16.04 | 85.4 | 98.2 |
| 06_calm3_w80t800 model_7199 | 12.68 | 88.5 | 0.123 | 11.22 | 5.6 | 0.087 | 99.2 | 98.2 | 15.88 | 86.1 | 98.2 |

## 06_calm3_t800_s1: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calm3_t800_s1 model_5900 | 12.67 | 86.9 | 0.145 | 11.01 | 6.9 | 0.133 | 98.9 | 97.1 | 15.97 | 84.4 | 96.3 |
| 06_calm3_t800_s1 model_6000 | 12.26 | 86.7 | 0.160 | 10.63 | 5.9 | 0.110 | 99.0 | 97.5 | 15.25 | 85.7 | 97.1 |
| 06_calm3_t800_s1 model_6100 | 12.34 | 88.1 | 0.133 | 10.87 | 5.1 | 0.095 | 99.1 | 98.2 | 15.53 | 84.6 | 97.5 |
| 06_calm3_t800_s1 model_6200 | 12.43 | 87.7 | 0.135 | 10.90 | 5.0 | 0.089 | 99.1 | 97.7 | 15.49 | 86.5 | 98.8 |
| 06_calm3_t800_s1 model_6300 | 12.57 | 88.1 | 0.137 | 11.07 | 4.9 | 0.083 | 99.0 | 98.6 | 15.69 | 85.4 | 97.9 |
| 06_calm3_t800_s1 model_6400 | 12.27 | 87.7 | 0.137 | 10.76 | 4.7 | 0.086 | 99.0 | 97.3 | 15.48 | 84.8 | 97.9 |
| 06_calm3_t800_s1 model_6500 | 12.74 | 88.7 | 0.129 | 11.30 | 4.8 | 0.082 | 99.0 | 98.0 | 15.69 | 86.5 | 99.0 |
| 06_calm3_t800_s1 model_6600 | 12.49 | 88.3 | 0.129 | 11.03 | 4.6 | 0.082 | 99.0 | 97.7 | 15.55 | 84.6 | 97.1 |
| 06_calm3_t800_s1 model_6700 | 12.26 | 88.3 | 0.139 | 10.82 | 4.5 | 0.080 | 99.0 | 97.5 | 15.63 | 86.5 | 98.4 |
| 06_calm3_t800_s1 model_6799 | 12.37 | 89.6 | 0.111 | 11.09 | 4.6 | 0.082 | 99.1 | 98.6 | 15.64 | 85.9 | 97.5 |

## 06_calmt_roll10: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.events.reset_robot_base.params.pose_range.roll=[-0.1745,0.1745]); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| reference: tilt800 model_6300 (rank 1), OWN on the roll ±10° task | 12.46 | 89.5 | 0.121 | 11.15 | 4.8 | 0.086 | 99.0 | 98.4 | 15.10 | 83.0 | 96.7 |
| 06_calmt_roll10 model_6400 | 12.49 | 89.3 | 0.119 | 11.15 | 4.8 | 0.079 | 99.0 | 97.7 | 15.28 | 84.0 | 98.4 |
| 06_calmt_roll10 model_6500 | 12.28 | 90.4 | 0.107 | 11.10 | 4.6 | 0.076 | 99.0 | 99.2 | 15.03 | 80.7 | 96.3 |
| 06_calmt_roll10 model_6600 | 12.39 | 85.4 | 0.168 | 10.57 | 4.6 | 0.080 | 99.0 | 98.0 | 15.14 | 80.5 | 97.7 |
| 06_calmt_roll10 model_6700 | 12.50 | 89.8 | 0.117 | 11.23 | 4.5 | 0.078 | 99.1 | 98.6 | 15.15 | 83.4 | 97.5 |
| 06_calmt_roll10 model_6800 | 12.68 | 86.5 | 0.145 | 10.97 | 4.6 | 0.082 | 99.1 | 98.0 | 15.08 | 81.6 | 98.0 |
| 06_calmt_roll10 model_6899 | 12.80 | 87.7 | 0.135 | 11.23 | 4.5 | 0.080 | 99.0 | 99.2 | 15.15 | 80.9 | 97.3 |

## 06_oneshot: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot model_4500 | 12.07 | 91.8 | 0.090 | 11.08 | 4.7 | 0.081 | 98.6 | 98.4 | 14.59 | 87.5 | 98.4 |
| 06_oneshot model_4600 | 12.49 | 89.6 | 0.109 | 11.20 | 4.5 | 0.080 | 98.7 | 98.0 | 14.74 | 88.3 | 98.6 |
| 06_oneshot model_4700 | 12.77 | 90.4 | 0.109 | 11.55 | 4.4 | 0.076 | 98.8 | 97.9 | 15.19 | 84.6 | 97.3 |
| 06_oneshot model_4800 | 12.78 | 90.2 | 0.117 | 11.54 | 4.2 | 0.074 | 99.0 | 98.6 | 15.37 | 86.1 | 99.0 |
| 06_oneshot model_4900 | 12.72 | 88.5 | 0.137 | 11.25 | 4.1 | 0.075 | 99.1 | 97.9 | 15.33 | 85.9 | 99.0 |
| 06_oneshot model_5000 | 12.92 | 90.4 | 0.102 | 11.68 | 4.2 | 0.074 | 99.1 | 99.2 | 15.58 | 85.9 | 99.0 |
| 06_oneshot model_5100 | 12.99 | 88.1 | 0.129 | 11.45 | 4.2 | 0.072 | 99.1 | 97.9 | 15.68 | 85.5 | 98.4 |
| 06_oneshot model_5200 | 13.23 | 88.3 | 0.145 | 11.68 | 4.2 | 0.073 | 99.1 | 97.9 | 15.59 | 85.7 | 98.0 |
| 06_oneshot model_5300 | 13.26 | 88.3 | 0.146 | 11.70 | 4.1 | 0.073 | 99.1 | 98.4 | 15.66 | 85.0 | 97.9 |
| 06_oneshot model_5400 | 13.07 | 89.5 | 0.121 | 11.70 | 4.1 | 0.074 | 98.9 | 98.8 | 15.69 | 85.2 | 97.9 |
| 06_oneshot model_5498 | 13.07 | 88.5 | 0.137 | 11.57 | 4.1 | 0.073 | 98.9 | 98.8 | 15.77 | 86.9 | 98.6 |

## 06_calmt_cmd2: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.commands.continuous_rotation.angular_vel_range=[0.0,0.0,2.0]); wrist speed over all 512 envs

| checkpoint | cmd | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_calmt_cmd2 model_6400 | 2.0 | 8.60 | 84.6 | 0.197 | 7.27 | 5.5 | 0.089 | 97.7 | 93.8 | 11.29 | 79.7 | 91.0 |
| 06_calmt_cmd2 model_6400 | 1.5 | 12.08 | 88.5 | 0.127 | 10.69 | 4.8 | 0.076 | 98.8 | 98.8 | - | - | - |
| 06_calmt_cmd2 model_6500 | 2.0 | 12.47 | 87.5 | 0.146 | 10.91 | 5.2 | 0.083 | 98.5 | 97.9 | 15.04 | 83.8 | 96.3 |
| 06_calmt_cmd2 model_6500 | 1.5 | 11.72 | 87.3 | 0.146 | 10.23 | 5.4 | 0.086 | 98.6 | 98.8 | - | - | - |
| 06_calmt_cmd2 model_6600 | 2.0 | 12.54 | 90.8 | 0.102 | 11.39 | 5.0 | 0.082 | 98.9 | 98.6 | 15.30 | 83.4 | 95.5 |
| 06_calmt_cmd2 model_6600 | 1.5 | 11.43 | 89.5 | 0.109 | 10.22 | 5.0 | 0.078 | 98.8 | 100.0 | - | - | - |
| 06_calmt_cmd2 model_6700 | 2.0 | 12.58 | 86.9 | 0.135 | 10.93 | 5.2 | 0.087 | 99.1 | 98.8 | 15.65 | 84.4 | 97.9 |
| 06_calmt_cmd2 model_6700 | 1.5 | 11.76 | 88.9 | 0.117 | 10.45 | 5.1 | 0.083 | 98.9 | 98.6 | - | - | - |
| 06_calmt_cmd2 model_6800 | 2.0 | 12.81 | 88.3 | 0.143 | 11.31 | 5.0 | 0.088 | 99.0 | 98.6 | 15.70 | 84.0 | 97.3 |
| 06_calmt_cmd2 model_6800 | 1.5 | 12.23 | 88.7 | 0.125 | 10.84 | 4.8 | 0.080 | 99.0 | 99.0 | - | - | - |
| 06_calmt_cmd2 model_6899 | 2.0 | 12.43 | 89.6 | 0.105 | 11.15 | 4.5 | 0.081 | 99.1 | 98.0 | 15.46 | 85.4 | 98.4 |
| 06_calmt_cmd2 model_6899 | 1.5 | 11.35 | 89.3 | 0.119 | 10.13 | 4.4 | 0.074 | 99.0 | 99.2 | - | - | - |

## rf2: 512-env eval on the PLA-skeleton friction (-realfric2-v0, its default DR), seed 42; wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| rank1_tilt800 2026-11-01_04-22-55_06_calm3_l1_tilt800_ft_r6k/model_6300 | 15.69 | 84.0 | 0.195 | 13.18 | 4.8 | 0.085 | 98.3 | 97.3 |
| reference_B_07rf2 2026-10-31_13-36-20_07_wp_rf2_ft_r6k/model_5200 | 14.20 | 85.5 | 0.174 | 12.15 | 6.0 | 0.078 | 90.2 | 95.3 |

## pos45s2_calmt: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pos45s2_calmt model_4500 | 10.42 | 89.3 | 0.123 | 9.30 | 3.9 | 0.077 | 97.1 | 94.9 | 13.52 | 86.9 | 95.7 |
| pos45s2_calmt model_4600 | 10.62 | 90.8 | 0.105 | 9.64 | 4.0 | 0.079 | 97.1 | 95.7 | 13.81 | 85.4 | 94.3 |
| pos45s2_calmt model_4700 | 11.62 | 91.6 | 0.104 | 10.64 | 4.0 | 0.078 | 97.0 | 97.5 | 14.20 | 87.1 | 96.1 |
| pos45s2_calmt model_4800 | 11.83 | 89.8 | 0.109 | 10.63 | 4.0 | 0.080 | 96.9 | 96.7 | 14.28 | 85.9 | 95.5 |
| pos45s2_calmt model_4900 | 11.89 | 91.2 | 0.098 | 10.84 | 4.0 | 0.079 | 96.0 | 96.5 | 14.28 | 87.7 | 97.3 |
| pos45s2_calmt model_5000 | 12.07 | 90.8 | 0.109 | 10.97 | 4.1 | 0.082 | 96.5 | 97.1 | 14.56 | 89.8 | 97.9 |
| pos45s2_calmt model_5100 | 12.03 | 89.6 | 0.113 | 10.78 | 4.0 | 0.079 | 96.9 | 95.5 | 14.56 | 87.7 | 97.7 |
| pos45s2_calmt model_5200 | 12.34 | 89.6 | 0.119 | 11.06 | 4.0 | 0.082 | 97.0 | 96.1 | 14.64 | 85.7 | 96.9 |
| pos45s2_calmt model_5300 | 12.50 | 88.1 | 0.145 | 11.01 | 4.1 | 0.080 | 97.1 | 95.9 | 14.58 | 86.5 | 97.9 |
| pos45s2_calmt model_5400 | 12.37 | 90.2 | 0.115 | 11.16 | 4.0 | 0.078 | 96.7 | 96.3 | 14.71 | 87.3 | 98.4 |
| pos45s2_calmt model_5498 | 12.56 | 91.4 | 0.105 | 11.48 | 4.0 | 0.078 | 96.7 | 97.3 | 14.74 | 87.3 | 97.7 |

## 07_calmt_pos45: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 07_calmt_pos45 model_4500 | 11.06 | 88.3 | 0.133 | 9.76 | 5.9 | 0.072 | 94.4 | 94.3 | 13.77 | 85.0 | 95.7 |
| 07_calmt_pos45 model_4600 | 10.99 | 86.9 | 0.143 | 9.56 | 5.8 | 0.071 | 94.9 | 94.7 | 13.62 | 85.2 | 95.3 |
| 07_calmt_pos45 model_4700 | 11.15 | 87.9 | 0.143 | 9.80 | 5.9 | 0.072 | 94.0 | 94.1 | 13.54 | 85.4 | 96.9 |
| 07_calmt_pos45 model_4800 | 11.24 | 86.7 | 0.162 | 9.74 | 5.8 | 0.072 | 94.6 | 93.2 | 13.64 | 84.0 | 95.3 |
| 07_calmt_pos45 model_4900 | 11.08 | 88.3 | 0.137 | 9.79 | 5.6 | 0.071 | 95.2 | 94.1 | 13.55 | 84.0 | 95.7 |
| 07_calmt_pos45 model_5000 | 11.22 | 87.7 | 0.135 | 9.84 | 5.7 | 0.070 | 95.0 | 94.5 | 13.47 | 83.2 | 94.7 |

## hist3_stage1: 512-env LOWFRIC on -upright-hist3-v0 + low-friction overrides, seed 42 (stage-2 start pick: best t>=2s, ties by rpm)

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|
| hist3_stage1 model_2500 | 10.16 | 77.3 | 0.303 | 7.85 | 6.6 | 74.5 | 80.5 |
| hist3_stage1 model_3500 | 10.53 | 74.2 | 0.365 | 7.81 | 6.3 | 78.5 | 77.1 |
| hist3_stage1 model_3999 | 10.65 | 72.7 | 0.379 | 7.74 | 6.1 | 82.7 | 76.6 |

## 06_oneshot_cont: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_cont model_5500 | 13.14 | 88.3 | 0.131 | 11.60 | 4.2 | 0.076 | 98.7 | 98.6 | 15.95 | 85.5 | 98.8 |
| 06_oneshot_cont model_5600 | 13.19 | 89.5 | 0.121 | 11.80 | 4.3 | 0.072 | 98.9 | 98.8 | 15.87 | 86.5 | 98.0 |

## 06_oneshot_l1w20: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.rewards.wrist_vel_l1.weight=-20.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
