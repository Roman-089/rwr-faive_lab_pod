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
