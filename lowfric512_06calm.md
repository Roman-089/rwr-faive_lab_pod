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
| 07_calmt_pos45 model_5100 | 11.26 | 88.1 | 0.125 | 9.92 | 5.6 | 0.070 | 94.9 | 94.5 | 13.63 | 85.4 | 97.5 |
| 07_calmt_pos45 model_5200 | 10.97 | 90.0 | 0.115 | 9.87 | 5.6 | 0.069 | 94.8 | 95.7 | 13.46 | 85.7 | 96.5 |
| 07_calmt_pos45 model_5300 | 10.83 | 88.5 | 0.125 | 9.58 | 5.6 | 0.069 | 95.1 | 94.9 | 13.30 | 85.0 | 97.3 |
| 07_calmt_pos45 model_5400 | 11.08 | 89.3 | 0.127 | 9.89 | 5.6 | 0.069 | 94.7 | 94.7 | 13.50 | 84.0 | 95.9 |
| 07_calmt_pos45 model_5499 | 11.06 | 89.1 | 0.129 | 9.85 | 5.7 | 0.068 | 93.2 | 93.9 | 13.37 | 85.2 | 97.1 |

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
| 06_oneshot_cont model_5700 | 13.25 | 89.3 | 0.129 | 11.83 | 4.2 | 0.070 | 99.1 | 98.8 | 15.92 | 87.3 | 98.4 |
| 06_oneshot_cont model_5800 | 13.25 | 87.7 | 0.137 | 11.62 | 4.1 | 0.072 | 99.0 | 98.6 | 15.77 | 85.2 | 98.0 |
| 06_oneshot_cont model_5900 | 13.26 | 86.5 | 0.152 | 11.47 | 4.4 | 0.076 | 98.8 | 99.2 | 15.84 | 85.2 | 98.6 |
| 06_oneshot_cont model_6000 | 13.50 | 90.6 | 0.113 | 12.24 | 4.5 | 0.074 | 98.9 | 98.8 | 16.04 | 84.8 | 98.0 |
| 06_oneshot_cont model_6100 | 13.26 | 88.3 | 0.141 | 11.70 | 4.3 | 0.073 | 98.9 | 98.8 | 15.99 | 84.0 | 98.4 |
| 06_oneshot_cont model_6200 | 13.41 | 88.3 | 0.146 | 11.84 | 4.2 | 0.072 | 99.0 | 98.6 | 15.83 | 85.5 | 98.4 |
| 06_oneshot_cont model_6300 | 13.19 | 86.5 | 0.158 | 11.41 | 4.2 | 0.077 | 98.9 | 98.0 | 15.78 | 85.0 | 97.5 |
| 06_oneshot_cont model_6400 | 13.36 | 86.1 | 0.160 | 11.51 | 4.3 | 0.078 | 99.0 | 98.8 | 15.92 | 85.0 | 98.6 |
| 06_oneshot_cont model_6497 | 12.75 | 83.6 | 0.213 | 10.66 | 4.5 | 0.084 | 98.8 | 96.9 | 14.73 | 85.5 | 98.6 |

## 06_oneshot_l1w20: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.rewards.wrist_vel_l1.weight=-20.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_l1w20 model_4500 | 11.85 | 90.4 | 0.113 | 10.71 | 4.7 | 0.091 | 98.8 | 97.1 | 14.28 | 83.8 | 97.1 |
| 06_oneshot_l1w20 model_4600 | 12.14 | 89.6 | 0.109 | 10.89 | 4.5 | 0.081 | 98.9 | 98.2 | 14.51 | 85.0 | 96.7 |
| 06_oneshot_l1w20 model_4700 | 12.23 | 90.2 | 0.109 | 11.04 | 4.6 | 0.084 | 98.7 | 97.9 | 14.58 | 85.0 | 97.5 |
| 06_oneshot_l1w20 model_4800 | 12.33 | 90.2 | 0.102 | 11.13 | 4.5 | 0.077 | 98.7 | 98.2 | 14.57 | 85.9 | 98.0 |
| 06_oneshot_l1w20 model_4900 | 12.29 | 91.0 | 0.096 | 11.19 | 4.3 | 0.076 | 98.8 | 98.4 | 14.75 | 86.1 | 97.5 |
| 06_oneshot_l1w20 model_5000 | 12.17 | 92.0 | 0.084 | 11.20 | 4.3 | 0.075 | 98.9 | 98.4 | 14.96 | 88.3 | 98.0 |
| 06_oneshot_l1w20 model_5100 | 12.30 | 92.2 | 0.084 | 11.34 | 4.2 | 0.074 | 98.8 | 98.0 | 14.93 | 85.2 | 98.0 |
| 06_oneshot_l1w20 model_5200 | 12.46 | 90.6 | 0.098 | 11.29 | 4.1 | 0.072 | 99.0 | 98.4 | 15.01 | 86.7 | 97.9 |
| 06_oneshot_l1w20 model_5300 | 12.49 | 90.4 | 0.102 | 11.30 | 4.2 | 0.072 | 98.8 | 99.2 | 15.08 | 86.5 | 97.7 |
| 06_oneshot_l1w20 model_5400 | 12.73 | 89.3 | 0.117 | 11.36 | 4.2 | 0.076 | 98.9 | 98.0 | 15.13 | 85.7 | 98.4 |
| 06_oneshot_l1w20 model_5498 | 12.81 | 89.6 | 0.105 | 11.49 | 4.2 | 0.076 | 99.0 | 99.4 | 15.23 | 85.9 | 98.2 |

## servo: 512-env LOWFRIC with the ORCA v2 servo model (--servo: delay 0-50 ms, nominal accel limit, play), seed 42; wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 | 9.95 | 40.8 | 1.471 | 4.06 | 7.5 | 0.149 | 96.3 | 76.8 |
| rank2_tilt800 2026-11-01_04-22-55_06_calm3_l1_tilt800_ft_r6k/model_6300 | 9.59 | 39.1 | 1.490 | 3.75 | 8.3 | 0.161 | 96.4 | 78.5 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 | 11.09 | 32.0 | 2.180 | 3.55 | 8.7 | 0.143 | 73.1 | 56.4 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-delay only | 15.42 | 89.8 | 0.119 | 13.85 | 4.1 | 0.074 | 98.9 | 98.2 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-delay only | 14.11 | 88.5 | 0.141 | 12.49 | 6.0 | 0.079 | 93.3 | 93.9 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-profile only | 10.29 | 52.3 | 0.959 | 5.39 | 6.5 | 0.164 | 97.0 | 78.5 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-profile only | 11.80 | 38.9 | 1.615 | 4.59 | 7.5 | 0.132 | 80.2 | 54.5 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-play only | 12.63 | 89.8 | 0.117 | 11.35 | 5.1 | 0.072 | 98.6 | 99.2 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-play only | 11.14 | 92.2 | 0.086 | 10.27 | 7.1 | 0.079 | 88.0 | 98.2 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-profile amax×1.5 | 13.19 | 75.0 | 0.395 | 9.89 | 5.3 | 0.120 | 98.3 | 90.8 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-profile amax×1.5 | 13.42 | 55.7 | 0.916 | 7.47 | 6.8 | 0.107 | 88.5 | 68.2 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-profile amax×2 | 14.75 | 81.4 | 0.244 | 12.01 | 4.6 | 0.097 | 98.4 | 94.7 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-profile amax×2 | 13.96 | 70.3 | 0.494 | 9.82 | 6.4 | 0.094 | 91.4 | 79.3 |
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 servo-profile amax×3 | 15.62 | 84.6 | 0.178 | 13.21 | 4.2 | 0.082 | 98.7 | 96.9 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 servo-profile amax×3 | 14.23 | 83.4 | 0.221 | 11.87 | 6.1 | 0.083 | 93.3 | 92.4 |

## 06_oneshot_s1: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_s1 model_4500 | 11.42 | 91.0 | 0.104 | 10.39 | 4.5 | 0.085 | 98.8 | 97.7 | 14.33 | 86.7 | 96.9 |
| 06_oneshot_s1 model_4600 | 11.80 | 90.6 | 0.100 | 10.70 | 4.4 | 0.080 | 98.8 | 98.0 | 14.61 | 86.1 | 97.5 |
| 06_oneshot_s1 model_4700 | 11.93 | 89.3 | 0.119 | 10.65 | 4.5 | 0.081 | 98.7 | 98.2 | 14.65 | 85.4 | 96.7 |
| 06_oneshot_s1 model_4800 | 12.45 | 91.8 | 0.086 | 11.42 | 4.4 | 0.076 | 98.9 | 98.6 | 14.92 | 86.1 | 97.7 |
| 06_oneshot_s1 model_4900 | 12.80 | 91.0 | 0.100 | 11.65 | 4.5 | 0.076 | 99.0 | 98.0 | 15.21 | 85.9 | 98.2 |
| 06_oneshot_s1 model_5000 | 12.59 | 92.4 | 0.078 | 11.63 | 4.2 | 0.073 | 99.1 | 99.0 | 15.14 | 86.5 | 98.2 |
| 06_oneshot_s1 model_5100 | 12.61 | 90.6 | 0.104 | 11.42 | 4.2 | 0.070 | 98.9 | 98.2 | 15.23 | 84.4 | 97.9 |
| 06_oneshot_s1 model_5200 | 12.75 | 90.4 | 0.100 | 11.53 | 4.2 | 0.073 | 99.1 | 97.9 | 15.30 | 85.0 | 98.2 |
| 06_oneshot_s1 model_5300 | 12.72 | 90.6 | 0.107 | 11.53 | 4.1 | 0.073 | 99.0 | 98.4 | 15.28 | 85.7 | 98.2 |
| 06_oneshot_s1 model_5400 | 12.81 | 90.2 | 0.100 | 11.56 | 4.1 | 0.074 | 99.1 | 98.6 | 15.27 | 85.9 | 98.0 |
| 06_oneshot_s1 model_5498 | 12.90 | 89.8 | 0.117 | 11.59 | 4.2 | 0.072 | 99.0 | 98.2 | 15.38 | 83.4 | 97.5 |

## 06_oneshot_track15: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_track15 model_4500 | 11.38 | 90.6 | 0.100 | 10.31 | 4.6 | 0.087 | 98.7 | 97.7 | 14.08 | 85.0 | 97.7 |
| 06_oneshot_track15 model_4600 | 11.99 | 92.6 | 0.084 | 11.10 | 4.4 | 0.081 | 98.9 | 98.4 | 14.32 | 84.8 | 97.5 |
| 06_oneshot_track15 model_4700 | 12.33 | 89.8 | 0.105 | 11.08 | 4.4 | 0.081 | 99.0 | 97.9 | 14.52 | 87.1 | 97.5 |
| 06_oneshot_track15 model_4800 | 12.38 | 91.6 | 0.088 | 11.34 | 4.4 | 0.081 | 98.9 | 96.5 | 14.51 | 85.9 | 97.7 |
| 06_oneshot_track15 model_4900 | 12.30 | 91.4 | 0.090 | 11.25 | 4.3 | 0.078 | 99.0 | 97.3 | 14.55 | 85.5 | 97.3 |
| 06_oneshot_track15 model_5000 | 12.28 | 92.4 | 0.092 | 11.34 | 4.2 | 0.076 | 99.0 | 98.0 | 14.58 | 86.1 | 97.9 |
| 06_oneshot_track15 model_5100 | 12.53 | 91.6 | 0.098 | 11.47 | 4.3 | 0.075 | 98.9 | 97.9 | 14.83 | 84.4 | 96.9 |
| 06_oneshot_track15 model_5200 | 12.35 | 89.6 | 0.115 | 11.08 | 4.3 | 0.076 | 99.0 | 97.1 | 14.83 | 84.8 | 96.7 |
| 06_oneshot_track15 model_5300 | 12.55 | 89.6 | 0.115 | 11.25 | 4.1 | 0.074 | 98.8 | 98.0 | 14.97 | 87.3 | 98.2 |
| 06_oneshot_track15 model_5400 | 12.55 | 90.8 | 0.104 | 11.40 | 4.3 | 0.075 | 98.8 | 97.9 | 15.11 | 87.5 | 99.2 |
| 06_oneshot_track15 model_5498 | 12.69 | 89.6 | 0.111 | 11.38 | 4.2 | 0.075 | 98.9 | 97.1 | 15.13 | 86.5 | 98.4 |

## hist3_stage2: 512-env LOWFRIC on -upright-hist3-v0 + low-friction overrides (stage 2 = -realfric-hist3 from stage-1 m2500, seed 42); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| hist3_stage2 model_3500 | 13.35 | 83.2 | 0.227 | 11.10 | 5.0 | 0.098 | 89.2 | 88.3 |
| hist3_stage2 model_3700 | 13.67 | 82.8 | 0.207 | 11.32 | 4.9 | 0.097 | 90.4 | 89.6 |
| hist3_stage2 model_3999 | 14.31 | 82.0 | 0.197 | 11.74 | 4.7 | 0.098 | 92.3 | 88.3 |

## 06_oneshot_servofast: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_servofast model_5500 | a no servo | 12.94 | 87.9 | 0.131 | 11.37 | 3.9 | 0.070 | 98.9 | 97.9 |
| 06_oneshot_servofast model_5500 | b profile x1.5 | 14.44 | 75.4 | 0.338 | 10.88 | 4.3 | 0.081 | 98.4 | 94.7 |
| 06_oneshot_servofast model_5500 | c full servo x1 | 11.64 | 46.5 | 1.184 | 5.41 | 6.2 | 0.101 | 96.9 | 86.7 |
| 06_oneshot_servofast model_5600 | a no servo | 12.98 | 85.9 | 0.174 | 11.16 | 4.2 | 0.072 | 98.8 | 96.3 |
| 06_oneshot_servofast model_5600 | b profile x1.5 | 14.51 | 74.2 | 0.365 | 10.77 | 4.3 | 0.078 | 98.3 | 95.3 |
| 06_oneshot_servofast model_5600 | c full servo x1 | 11.97 | 41.6 | 1.291 | 4.98 | 6.0 | 0.097 | 96.8 | 82.6 |
| 06_oneshot_servofast model_5700 | a no servo | 12.69 | 88.7 | 0.129 | 11.25 | 4.2 | 0.075 | 98.7 | 97.7 |
| 06_oneshot_servofast model_5700 | b profile x1.5 | 14.64 | 77.1 | 0.307 | 11.29 | 4.2 | 0.076 | 98.3 | 95.3 |
| 06_oneshot_servofast model_5700 | c full servo x1 | 12.20 | 46.1 | 1.225 | 5.62 | 5.8 | 0.093 | 97.0 | 85.5 |
| 06_oneshot_servofast model_5800 | a no servo | 12.68 | 88.9 | 0.125 | 11.27 | 4.1 | 0.073 | 99.0 | 99.2 |
| 06_oneshot_servofast model_5800 | b profile x1.5 | 14.56 | 77.5 | 0.297 | 11.29 | 4.2 | 0.078 | 98.2 | 95.7 |
| 06_oneshot_servofast model_5800 | c full servo x1 | 12.23 | 45.1 | 1.266 | 5.52 | 5.8 | 0.091 | 96.9 | 84.4 |
| 06_oneshot_servofast model_5900 | a no servo | 12.84 | 87.9 | 0.141 | 11.29 | 4.1 | 0.074 | 98.9 | 97.5 |
| 06_oneshot_servofast model_5900 | b profile x1.5 | 14.77 | 76.4 | 0.332 | 11.28 | 4.2 | 0.077 | 98.1 | 95.1 |
| 06_oneshot_servofast model_5900 | c full servo x1 | 12.37 | 45.3 | 1.201 | 5.61 | 5.7 | 0.089 | 97.0 | 85.9 |
| 06_oneshot_servofast model_5999 | a no servo | 12.62 | 87.7 | 0.146 | 11.07 | 4.3 | 0.080 | 98.7 | 96.7 |
| 06_oneshot_servofast model_5999 | b profile x1.5 | 14.70 | 77.7 | 0.309 | 11.43 | 4.1 | 0.078 | 98.3 | 96.7 |
| 06_oneshot_servofast model_5999 | c full servo x1 | 12.41 | 44.3 | 1.205 | 5.50 | 5.8 | 0.094 | 96.9 | 87.5 |

## hist3_calmt: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-hist3-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hist3_calmt model_4000 | 15.45 | 74.8 | 0.330 | 11.55 | 5.1 | 0.102 | 89.0 | 84.6 | n/a | n/a | n/a |
| hist3_calmt model_4100 | 15.89 | 77.7 | 0.285 | 12.35 | 4.9 | 0.100 | 90.6 | 87.5 | n/a | n/a | n/a |
| hist3_calmt model_4200 | 15.48 | 76.8 | 0.301 | 11.88 | 4.8 | 0.099 | 90.6 | 86.7 | n/a | n/a | n/a |
| hist3_calmt model_4300 | 15.48 | 79.3 | 0.295 | 12.28 | 4.8 | 0.096 | 91.6 | 88.1 | n/a | n/a | n/a |
| hist3_calmt model_4400 | 15.82 | 77.7 | 0.303 | 12.30 | 4.7 | 0.099 | 92.4 | 86.9 | n/a | n/a | n/a |
| hist3_calmt model_4500 | 15.84 | 79.7 | 0.260 | 12.62 | 4.6 | 0.094 | 93.7 | 89.1 | n/a | n/a | n/a |
| hist3_calmt model_4600 | 15.84 | 80.7 | 0.230 | 12.77 | 4.6 | 0.092 | 93.7 | 90.6 | n/a | n/a | n/a |
| hist3_calmt model_4700 | 15.89 | 80.9 | 0.254 | 12.85 | 4.6 | 0.091 | 93.0 | 89.6 | n/a | n/a | n/a |
| hist3_calmt model_4800 | 15.97 | 78.3 | 0.285 | 12.51 | 4.5 | 0.090 | 93.4 | 89.8 | n/a | n/a | n/a |
| hist3_calmt model_4900 | 15.94 | 78.7 | 0.291 | 12.55 | 4.6 | 0.091 | 93.3 | 89.3 | n/a | n/a | n/a |
| hist3_calmt model_5000 | 16.20 | 78.9 | 0.281 | 12.78 | 4.5 | 0.092 | 94.6 | 88.9 | n/a | n/a | n/a |
| hist3_calmt model_5100 | 15.84 | 79.5 | 0.264 | 12.59 | 4.5 | 0.091 | 92.6 | 90.8 | n/a | n/a | n/a |
| hist3_calmt model_5200 | 15.95 | 79.1 | 0.260 | 12.61 | 4.5 | 0.091 | 94.7 | 89.8 | n/a | n/a | n/a |
| hist3_calmt model_5300 | 15.80 | 81.8 | 0.215 | 12.93 | 4.5 | 0.090 | 93.7 | 91.4 | n/a | n/a | n/a |
| hist3_calmt model_5400 | 16.24 | 80.1 | 0.232 | 13.01 | 4.4 | 0.091 | 94.7 | 91.2 | n/a | n/a | n/a |
| hist3_calmt model_5498 | 16.06 | 79.1 | 0.266 | 12.70 | 4.5 | 0.092 | 93.3 | 89.5 | n/a | n/a | n/a |

## placed: 512-env LOWFRIC with a placed spawn (cube starts flat: reset roll = pitch = 0, random yaw), seed 42; wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| rank1_06_oneshot 2026-11-01_06-38-47_06_oneshot_calmt_ft_r6k/model_5400 | 13.09 | 90.2 | 0.109 | 11.81 | 4.0 | 0.072 | 99.0 | 98.2 |
| A_07_wp_rf_ft 2026-10-31_13-07-57_07_wp_rf_ft_r6k/model_4600 | 11.38 | 91.2 | 0.100 | 10.38 | 6.1 | 0.078 | 92.7 | 95.5 |

## 06_oneshot_servofast_s1: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_servofast_s1 model_5500 | a no servo | 12.98 | 91.2 | 0.100 | 11.84 | 4.0 | 0.067 | 99.0 | 99.2 |
| 06_oneshot_servofast_s1 model_5500 | b profile x1.5 | 14.07 | 77.5 | 0.285 | 10.91 | 4.3 | 0.080 | 98.5 | 95.9 |
| 06_oneshot_servofast_s1 model_5500 | c full servo x1 | 11.31 | 40.8 | 1.381 | 4.61 | 6.4 | 0.110 | 96.7 | 80.7 |
| 06_oneshot_servofast_s1 model_5600 | a no servo | 12.78 | 89.8 | 0.115 | 11.49 | 4.2 | 0.068 | 99.0 | 98.8 |
| 06_oneshot_servofast_s1 model_5600 | b profile x1.5 | 14.21 | 76.6 | 0.322 | 10.88 | 4.3 | 0.078 | 98.3 | 94.1 |
| 06_oneshot_servofast_s1 model_5600 | c full servo x1 | 11.73 | 46.1 | 1.100 | 5.41 | 6.1 | 0.096 | 97.2 | 87.1 |
| 06_oneshot_servofast_s1 model_5700 | a no servo | 12.71 | 88.9 | 0.129 | 11.30 | 4.2 | 0.072 | 98.9 | 98.6 |
| 06_oneshot_servofast_s1 model_5700 | b profile x1.5 | 14.44 | 78.7 | 0.295 | 11.36 | 4.3 | 0.077 | 98.1 | 94.5 |
| 06_oneshot_servofast_s1 model_5700 | c full servo x1 | 11.93 | 46.7 | 1.164 | 5.57 | 6.1 | 0.095 | 97.2 | 86.9 |
| 06_oneshot_servofast_s1 model_5800 | a no servo | 12.77 | 90.0 | 0.115 | 11.50 | 4.1 | 0.067 | 99.0 | 99.2 |
| 06_oneshot_servofast_s1 model_5800 | b profile x1.5 | 14.43 | 77.5 | 0.303 | 11.19 | 4.2 | 0.074 | 98.3 | 95.9 |
| 06_oneshot_servofast_s1 model_5800 | c full servo x1 | 11.90 | 45.3 | 1.201 | 5.39 | 5.9 | 0.091 | 97.1 | 85.2 |
| 06_oneshot_servofast_s1 model_5900 | a no servo | 12.41 | 88.1 | 0.141 | 10.93 | 4.1 | 0.068 | 98.9 | 98.4 |
| 06_oneshot_servofast_s1 model_5900 | b profile x1.5 | 14.39 | 78.7 | 0.312 | 11.32 | 4.1 | 0.073 | 98.4 | 96.7 |
| 06_oneshot_servofast_s1 model_5900 | c full servo x1 | 11.91 | 46.7 | 1.188 | 5.56 | 5.8 | 0.093 | 97.2 | 85.4 |
| 06_oneshot_servofast_s1 model_5999 | a no servo | 12.37 | 87.7 | 0.146 | 10.85 | 4.0 | 0.067 | 98.9 | 99.0 |
| 06_oneshot_servofast_s1 model_5999 | b profile x1.5 | 14.45 | 77.5 | 0.281 | 11.20 | 4.1 | 0.072 | 98.2 | 96.5 |
| 06_oneshot_servofast_s1 model_5999 | c full servo x1 | 12.04 | 49.0 | 1.039 | 5.90 | 5.7 | 0.089 | 97.5 | 88.1 |

## 06_servofast_from_base: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| 06_servofast_from_base model_4500 | a no servo | 7.40 | 88.7 | 0.135 | 6.56 | 5.0 | 0.089 | 97.8 | 97.1 |
| 06_servofast_from_base model_4500 | b profile x1.5 | 12.38 | 75.0 | 0.357 | 9.29 | 5.3 | 0.094 | 97.6 | 94.5 |
| 06_servofast_from_base model_4500 | c full servo x1 | 10.86 | 42.0 | 1.447 | 4.56 | 7.4 | 0.115 | 96.8 | 81.4 |
| 06_servofast_from_base model_4600 | a no servo | 8.28 | 91.2 | 0.088 | 7.55 | 5.0 | 0.094 | 98.2 | 96.7 |
| 06_servofast_from_base model_4600 | b profile x1.5 | 13.01 | 75.4 | 0.371 | 9.81 | 4.9 | 0.089 | 98.1 | 95.3 |
| 06_servofast_from_base model_4600 | c full servo x1 | 11.59 | 42.8 | 1.492 | 4.96 | 6.7 | 0.105 | 96.7 | 81.1 |
| 06_servofast_from_base model_4700 | a no servo | 9.18 | 92.0 | 0.088 | 8.44 | 4.7 | 0.084 | 98.5 | 97.9 |
| 06_servofast_from_base model_4700 | b profile x1.5 | 13.55 | 75.2 | 0.352 | 10.19 | 4.7 | 0.083 | 98.0 | 94.7 |
| 06_servofast_from_base model_4700 | c full servo x1 | 11.77 | 43.4 | 1.400 | 5.10 | 6.5 | 0.100 | 97.0 | 82.4 |
| 06_servofast_from_base model_4800 | a no servo | 9.90 | 91.2 | 0.104 | 9.03 | 4.5 | 0.077 | 98.2 | 98.2 |
| 06_servofast_from_base model_4800 | b profile x1.5 | 13.90 | 76.2 | 0.326 | 10.59 | 4.4 | 0.078 | 98.2 | 94.5 |
| 06_servofast_from_base model_4800 | c full servo x1 | 11.96 | 41.8 | 1.406 | 5.00 | 6.2 | 0.096 | 97.1 | 82.0 |
| 06_servofast_from_base model_4900 | a no servo | 10.17 | 91.2 | 0.098 | 9.27 | 4.7 | 0.088 | 98.3 | 98.4 |
| 06_servofast_from_base model_4900 | b profile x1.5 | 14.16 | 74.6 | 0.355 | 10.57 | 4.4 | 0.079 | 98.4 | 95.5 |
| 06_servofast_from_base model_4900 | c full servo x1 | 12.05 | 45.5 | 1.248 | 5.48 | 6.2 | 0.096 | 97.2 | 87.9 |

## 06_oneshot_servowide: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| 06_oneshot_servowide model_5500 | a no servo | 12.68 | 88.9 | 0.135 | 11.26 | 4.0 | 0.067 | 99.0 | 98.2 |
| 06_oneshot_servowide model_5500 | b profile x1.5 | 14.29 | 80.1 | 0.279 | 11.45 | 4.3 | 0.078 | 98.5 | 97.5 |
| 06_oneshot_servowide model_5500 | c full servo x1 | 11.92 | 44.1 | 1.236 | 5.26 | 6.1 | 0.100 | 97.2 | 84.8 |
| 06_oneshot_servowide model_5600 | a no servo | 12.27 | 88.1 | 0.139 | 10.81 | 4.0 | 0.068 | 99.1 | 98.2 |
| 06_oneshot_servowide model_5600 | b profile x1.5 | 14.30 | 79.7 | 0.293 | 11.39 | 4.2 | 0.075 | 98.5 | 96.1 |
| 06_oneshot_servowide model_5600 | c full servo x1 | 12.21 | 44.5 | 1.197 | 5.44 | 5.6 | 0.085 | 97.3 | 87.3 |
| 06_oneshot_servowide model_5700 | a no servo | 12.54 | 88.3 | 0.141 | 11.07 | 4.2 | 0.072 | 98.8 | 98.2 |
| 06_oneshot_servowide model_5700 | b profile x1.5 | 14.68 | 77.5 | 0.312 | 11.38 | 4.1 | 0.071 | 98.5 | 96.3 |
| 06_oneshot_servowide model_5700 | c full servo x1 | 12.51 | 43.0 | 1.227 | 5.37 | 5.6 | 0.084 | 97.3 | 83.6 |
| 06_oneshot_servowide model_5800 | a no servo | 12.75 | 89.6 | 0.123 | 11.43 | 4.1 | 0.067 | 98.9 | 98.8 |
| 06_oneshot_servowide model_5800 | b profile x1.5 | 14.79 | 76.6 | 0.328 | 11.32 | 4.1 | 0.072 | 98.1 | 94.9 |
| 06_oneshot_servowide model_5800 | c full servo x1 | 12.62 | 47.7 | 1.139 | 6.01 | 5.5 | 0.078 | 97.5 | 88.3 |
| 06_oneshot_servowide model_5900 | a no servo | 12.73 | 87.7 | 0.156 | 11.17 | 4.2 | 0.070 | 98.8 | 97.3 |
| 06_oneshot_servowide model_5900 | b profile x1.5 | 14.69 | 76.0 | 0.322 | 11.16 | 4.1 | 0.070 | 98.4 | 95.3 |
| 06_oneshot_servowide model_5900 | c full servo x1 | 12.54 | 47.5 | 1.127 | 5.95 | 5.5 | 0.079 | 97.5 | 89.1 |
| 06_oneshot_servowide model_5999 | a no servo | 12.47 | 88.7 | 0.131 | 11.05 | 4.0 | 0.072 | 98.9 | 98.0 |
| 06_oneshot_servowide model_5999 | b profile x1.5 | 14.66 | 76.4 | 0.338 | 11.20 | 3.9 | 0.071 | 98.2 | 95.3 |
| 06_oneshot_servowide model_5999 | c full servo x1 | 12.66 | 47.3 | 1.072 | 5.98 | 5.3 | 0.080 | 97.4 | 90.2 |

## hist3_calmt_servo: every checkpoint, 512-env LOWFRIC (a) no servo, (c) full servo model x1, on the hist3 task + friction overrides; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| hist3_calmt_servo model_4000 | a no servo | 15.86 | 77.1 | 0.311 | 12.23 | 4.9 | 0.103 | 90.0 | 86.5 |
| hist3_calmt_servo model_4000 | c full servo x1 | 8.30 | 26.6 | 2.506 | 2.20 | 9.8 | 0.153 | 64.8 | 44.7 |
| hist3_calmt_servo model_4100 | a no servo | 15.68 | 77.9 | 0.283 | 12.22 | 4.6 | 0.096 | 92.1 | 87.9 |
| hist3_calmt_servo model_4100 | c full servo x1 | 9.63 | 34.8 | 1.830 | 3.35 | 8.4 | 0.138 | 73.0 | 59.4 |
| hist3_calmt_servo model_4200 | a no servo | 15.30 | 77.1 | 0.268 | 11.81 | 4.7 | 0.092 | 92.3 | 88.5 |
| hist3_calmt_servo model_4200 | c full servo x1 | 10.58 | 35.7 | 1.629 | 3.78 | 7.5 | 0.131 | 79.3 | 65.8 |
| hist3_calmt_servo model_4300 | a no servo | 15.16 | 79.9 | 0.277 | 12.11 | 4.4 | 0.086 | 93.6 | 88.5 |
| hist3_calmt_servo model_4300 | c full servo x1 | 11.18 | 44.7 | 1.289 | 5.00 | 6.9 | 0.122 | 83.9 | 75.2 |
| hist3_calmt_servo model_4400 | a no servo | 14.90 | 80.5 | 0.227 | 11.99 | 4.3 | 0.086 | 93.1 | 92.4 |
| hist3_calmt_servo model_4400 | c full servo x1 | 11.31 | 43.9 | 1.273 | 4.97 | 6.6 | 0.120 | 83.8 | 73.8 |

## 06_cont_servofast: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
| 06_cont_servofast model_6100 | a no servo | 13.29 | 88.5 | 0.137 | 11.76 | 4.3 | 0.071 | 98.9 | 98.6 |
| 06_cont_servofast model_6100 | b profile x1.5 | 14.80 | 76.2 | 0.336 | 11.27 | 4.5 | 0.080 | 98.0 | 94.5 |
| 06_cont_servofast model_6100 | c full servo x1 | 12.17 | 44.5 | 1.297 | 5.42 | 6.3 | 0.098 | 96.7 | 84.4 |
| 06_cont_servofast model_6200 | a no servo | 13.05 | 90.2 | 0.113 | 11.77 | 4.1 | 0.068 | 98.8 | 98.2 |
| 06_cont_servofast model_6200 | b profile x1.5 | 14.82 | 78.7 | 0.305 | 11.66 | 4.2 | 0.075 | 98.2 | 96.1 |
| 06_cont_servofast model_6200 | c full servo x1 | 12.46 | 45.3 | 1.162 | 5.64 | 5.9 | 0.090 | 96.8 | 87.9 |

## 06_cont_servowide: every checkpoint, 512-env LOWFRIC (a) no servo, (b) servo profile only at amax x1.5, (c) full servo model x1; wrist speed over all 512 envs

| checkpoint | eval | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|---|
