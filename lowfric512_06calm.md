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

## 06_oneshot: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0); wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|

## 06_calmt_cmd2: every checkpoint, 512-env LOWFRIC + 512-env OWN (orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-v0, env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0 env.commands.continuous_rotation.angular_vel_range=[0.0,0.0,2.0]); wrist speed over all 512 envs

| checkpoint | cmd | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % | OWN512 rpm | OWN512 drop-free % | OWN512 t>=2s % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## rf2: 512-env eval on the PLA-skeleton friction (-realfric2-v0, its default DR), seed 42; wrist speed over all 512 envs

| checkpoint | rpm | drop-free % | drops/min | score | tilt deg | wrist speed | in region % | drop-free t>=2s % |
|---|---|---|---|---|---|---|---|---|
| rank1_tilt800 2026-11-01_04-22-55_06_calm3_l1_tilt800_ft_r6k/model_6300 | 15.69 | 84.0 | 0.195 | 13.18 | 4.8 | 0.085 | 98.3 | 97.3 |
| reference_B_07rf2 2026-10-31_13-36-20_07_wp_rf2_ft_r6k/model_5200 | 14.20 | 85.5 | 0.174 | 12.15 | 6.0 | 0.078 | 90.2 | 95.3 |
