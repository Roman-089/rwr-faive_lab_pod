# Per-env RPM spread, 512-env LOWFRIC (2026-11-01T22:22Z)

Each env is one draw of the randomised friction / mass, like one hand trial. Per-env RPM over the alive steps at t >= 2 s (cube_motion --spread). stall = envs below 5 RPM; good trial = no drop at t >= 2 s and >= 8 RPM. Seed 42, 512 envs, -upright-v0 + low-friction overrides (hist3 rows: -wristl1-hist3 task). Rank-1 servofast rows were re-run for this (--tag _lowfric512spr) because the 3-way watcher had deleted its npz.


## rank 1

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| servofast m5800 | (a) no servo | 12.68 | 8.88 | 13.74 | 5.9 | 91.2 | 99.2 | 99.0 |
| servofast m5800 | (b) profile x1.5 | 14.56 | 11.96 | 15.56 | 2.7 | 90.6 | 95.7 | 98.2 |
| servofast m5800 | (c) full servo x1 | 12.23 | 10.21 | 13.09 | 3.7 | 79.3 | 84.4 | 96.9 |

## references

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| rank 2 06_oneshot m5400 | LOWFRIC | 13.07 | 8.54 | 14.33 | 6.4 | 89.6 | 98.8 | 98.9 |
| 06_oneshot m5400 | (b) profile x1.5 | 13.19 | 7.94 | 14.67 | 6.6 | 82.2 | 90.8 | 98.3 |
| 06_oneshot m5400 | (c) full servo x1 | 9.95 | 5.66 | 10.71 | 7.8 | 64.1 | 76.8 | 96.3 |
| tilt800 m6300 | LOWFRIC | 12.46 | 7.74 | 13.86 | 7.2 | 87.5 | 98.4 | 99.0 |
| calm3 m7000 | LOWFRIC | 12.93 | 9.78 | 13.85 | 3.9 | 92.2 | 98.2 | 98.8 |
| A 07_wp_rf_ft m4600 | LOWFRIC | 11.41 | 7.61 | 12.42 | 4.3 | 84.8 | 96.5 | 93.2 |
| 07ft seed1 m4900 | LOWFRIC | 11.39 | 7.25 | 12.40 | 4.1 | 85.2 | 98.0 | 94.2 |
| region200c m4800 | LOWFRIC | 11.48 | 7.84 | 12.42 | 3.3 | 86.9 | 96.9 | 94.7 |

## others

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| oneshot_cont m6000 | LOWFRIC | 13.50 | 11.28 | 14.25 | 3.5 | 94.5 | 98.8 | 98.9 |

## servo DR runs

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| servofast_s1 m5500 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5500 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5500 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5600 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5600 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5600 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5700 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5700 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5700 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5800 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5800 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5800 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servofast_s1 m5900 | (a) no servo | 12.41 | 7.76 | 13.60 | 6.8 | 87.9 | 98.4 | 98.9 |
| servofast_s1 m5900 | (b) profile x1.5 | 14.39 | 12.12 | 15.49 | 3.5 | 90.6 | 96.7 | 98.4 |
| servofast_s1 m5900 | (c) full servo x1 | 11.91 | 10.02 | 12.82 | 4.1 | 80.3 | 85.4 | 97.2 |
| servofast_s1 m5999 | (a) no servo | 12.37 | 8.63 | 13.48 | 7.0 | 89.8 | 99.0 | 98.9 |
| servofast_s1 m5999 | (b) profile x1.5 | 14.45 | 12.09 | 15.43 | 3.7 | 91.8 | 96.5 | 98.2 |
| servofast_s1 m5999 | (c) full servo x1 | 12.04 | 10.34 | 12.83 | 2.3 | 83.4 | 88.1 | 97.5 |
| servowide m5500 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servowide m5500 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servowide m5500 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servowide m5600 | (a) no servo | n/a (512 npz deleted) | | | | | | |
| servowide m5600 | (b) profile x1.5 | n/a (512 npz deleted) | | | | | | |
| servowide m5600 | (c) full servo x1 | n/a (512 npz deleted) | | | | | | |
| servowide m5700 | (a) no servo | 12.54 | 8.32 | 13.83 | 6.2 | 88.3 | 98.2 | 98.8 |
| servowide m5700 | (b) profile x1.5 | 14.68 | 12.38 | 15.65 | 4.1 | 91.4 | 96.3 | 98.5 |
| servowide m5700 | (c) full servo x1 | 12.51 | 10.65 | 13.25 | 2.7 | 79.9 | 83.6 | 97.3 |
| servowide m5800 | (a) no servo | 12.75 | 9.25 | 13.79 | 5.5 | 90.2 | 98.8 | 98.9 |
| servowide m5800 | (b) profile x1.5 | 14.79 | 12.65 | 15.60 | 2.7 | 90.8 | 94.9 | 98.1 |
| servowide m5800 | (c) full servo x1 | 12.62 | 11.32 | 13.34 | 3.1 | 83.8 | 88.3 | 97.5 |
| servowide m5900 | (a) no servo | 12.73 | 8.34 | 13.80 | 5.3 | 88.5 | 97.3 | 98.8 |
| servowide m5900 | (b) profile x1.5 | 14.69 | 12.61 | 15.58 | 3.7 | 90.8 | 95.3 | 98.4 |
| servowide m5900 | (c) full servo x1 | 12.54 | 10.97 | 13.21 | 2.9 | 85.0 | 89.1 | 97.5 |
| servowide m5999 | (a) no servo | 12.47 | 8.36 | 13.75 | 6.6 | 88.7 | 98.0 | 98.9 |
| servowide m5999 | (b) profile x1.5 | 14.66 | 12.19 | 15.63 | 3.1 | 90.0 | 95.3 | 98.2 |
| servowide m5999 | (c) full servo x1 | 12.66 | 11.25 | 13.38 | 3.1 | 86.3 | 90.2 | 97.4 |
| servofast_from_base m4500 | (a) no servo | 7.40 | 0.43 | 7.69 | 36.7 | 46.1 | 97.1 | 97.8 |
| servofast_from_base m4500 | (b) profile x1.5 | 12.38 | 3.82 | 15.04 | 13.3 | 73.8 | 94.5 | 97.6 |
| servofast_from_base m4500 | (c) full servo x1 | 10.86 | 6.32 | 12.03 | 7.6 | 71.9 | 81.4 | 96.8 |
| servofast_from_base m4600 | (a) no servo | 8.28 | 0.52 | 8.46 | 29.3 | 50.2 | 96.7 | 98.2 |
| servofast_from_base m4600 | (b) profile x1.5 | 13.01 | 4.60 | 15.29 | 10.5 | 79.7 | 95.3 | 98.1 |
| servofast_from_base m4600 | (c) full servo x1 | 11.59 | 8.66 | 12.59 | 5.9 | 73.8 | 81.1 | 96.7 |
| servofast_from_base m4700 | (a) no servo | 9.18 | 0.59 | 10.06 | 24.0 | 60.0 | 97.9 | 98.5 |
| servofast_from_base m4700 | (b) profile x1.5 | 13.55 | 6.93 | 15.61 | 7.6 | 83.2 | 94.7 | 98.0 |
| servofast_from_base m4700 | (c) full servo x1 | 11.77 | 9.06 | 12.72 | 4.3 | 75.8 | 82.4 | 97.0 |
| servofast_from_base m4800 | (a) no servo | 9.90 | 1.63 | 10.55 | 18.4 | 66.8 | 98.2 | 98.2 |
| servofast_from_base m4800 | (b) profile x1.5 | 13.90 | 8.39 | 15.61 | 6.1 | 85.9 | 94.5 | 98.2 |
| servofast_from_base m4800 | (c) full servo x1 | 11.96 | 9.72 | 12.79 | 3.7 | 77.1 | 82.0 | 97.1 |
| servofast_from_base m4900 | (a) no servo | 10.17 | 2.00 | 11.13 | 16.0 | 68.0 | 98.4 | 98.3 |
| servofast_from_base m4900 | (b) profile x1.5 | 14.16 | 9.72 | 15.79 | 6.1 | 87.1 | 95.5 | 98.4 |
| servofast_from_base m4900 | (c) full servo x1 | 12.05 | 9.98 | 12.90 | 3.7 | 83.4 | 87.9 | 97.2 |
| servofast_from_base m5000 | (a) no servo | 10.78 | 3.08 | 11.99 | 14.1 | 74.2 | 98.4 | 98.1 |
| servofast_from_base m5000 | (b) profile x1.5 | 14.33 | 10.48 | 15.74 | 4.9 | 87.5 | 95.1 | 98.3 |
| servofast_from_base m5000 | (c) full servo x1 | 12.11 | 10.62 | 12.84 | 2.0 | 78.3 | 82.2 | 97.2 |
| servofast_from_base m5100 | (a) no servo | 10.54 | 2.53 | 12.00 | 14.8 | 71.1 | 97.1 | 98.3 |
| servofast_from_base m5100 | (b) profile x1.5 | 14.15 | 9.81 | 15.66 | 5.9 | 87.1 | 94.9 | 98.5 |
| servofast_from_base m5100 | (c) full servo x1 | 12.18 | 10.71 | 12.91 | 2.5 | 82.0 | 86.3 | 97.4 |
| servofast_from_base m5200 | (a) no servo | 11.27 | 3.58 | 13.05 | 12.7 | 76.4 | 96.5 | 98.4 |
| servofast_from_base m5200 | (b) profile x1.5 | 14.37 | 11.11 | 15.65 | 5.3 | 89.6 | 96.7 | 98.3 |
| servofast_from_base m5200 | (c) full servo x1 | 12.26 | 11.09 | 12.96 | 3.1 | 83.2 | 87.1 | 97.3 |
| servofast_from_base m5300 | (a) no servo | 11.51 | 3.98 | 13.20 | 11.5 | 78.1 | 96.7 | 98.0 |
| servofast_from_base m5300 | (b) profile x1.5 | 14.70 | 11.97 | 15.89 | 3.7 | 91.2 | 96.3 | 98.5 |
| servofast_from_base m5300 | (c) full servo x1 | 12.30 | 10.51 | 12.97 | 1.6 | 82.2 | 85.4 | 97.0 |
| servofast_from_base m5400 | (a) no servo | 11.42 | 3.80 | 13.17 | 12.1 | 77.0 | 97.5 | 98.0 |
| servofast_from_base m5400 | (b) profile x1.5 | 14.70 | 12.50 | 15.77 | 3.9 | 92.4 | 97.3 | 98.4 |
| servofast_from_base m5400 | (c) full servo x1 | 12.22 | 10.69 | 12.91 | 2.5 | 83.2 | 87.1 | 97.1 |
| servofast_from_base m5498 | (a) no servo | 11.74 | 5.40 | 13.27 | 9.6 | 80.7 | 96.7 | 97.9 |
| servofast_from_base m5498 | (b) profile x1.5 | 14.71 | 11.97 | 15.84 | 4.1 | 91.4 | 96.7 | 98.3 |
| servofast_from_base m5498 | (c) full servo x1 | 12.28 | 10.56 | 12.89 | 1.6 | 82.4 | 85.7 | 97.1 |
| 06_cont_servofast m6100 | (a) no servo | 13.29 | 11.29 | 14.05 | 2.9 | 93.8 | 98.6 | 98.9 |
| 06_cont_servofast m6100 | (b) profile x1.5 | 14.80 | 12.51 | 15.66 | 3.3 | 90.6 | 94.5 | 98.0 |
| 06_cont_servofast m6100 | (c) full servo x1 | 12.17 | 9.75 | 13.04 | 2.7 | 80.3 | 84.4 | 96.7 |
| 06_cont_servofast m6200 | (a) no servo | 13.05 | 10.06 | 14.05 | 4.1 | 91.8 | 98.2 | 98.8 |
| 06_cont_servofast m6200 | (b) profile x1.5 | 14.82 | 12.10 | 15.83 | 2.9 | 91.8 | 96.1 | 98.2 |
| 06_cont_servofast m6200 | (c) full servo x1 | 12.46 | 10.25 | 13.30 | 2.9 | 84.0 | 87.9 | 96.8 |
| 06_cont_servofast m6300 | (a) no servo | 13.08 | 9.68 | 14.06 | 4.9 | 90.4 | 98.2 | 98.9 |
| 06_cont_servofast m6300 | (b) profile x1.5 | 14.88 | 12.74 | 15.81 | 3.5 | 89.6 | 94.3 | 98.2 |
| 06_cont_servofast m6300 | (c) full servo x1 | 12.58 | 10.72 | 13.31 | 1.8 | 83.6 | 87.7 | 96.7 |
| 06_cont_servofast m6399 | (a) no servo | 13.18 | 10.91 | 14.17 | 3.7 | 91.4 | 97.3 | 98.6 |
| 06_cont_servofast m6399 | (b) profile x1.5 | 14.98 | 12.96 | 15.87 | 2.9 | 93.4 | 97.3 | 98.2 |
| 06_cont_servofast m6399 | (c) full servo x1 | 12.51 | 10.72 | 13.36 | 2.7 | 83.4 | 86.5 | 96.7 |
| 06_cont_servowide m6100 | (a) no servo | 13.27 | 11.47 | 13.97 | 3.5 | 93.9 | 98.0 | 98.9 |
| 06_cont_servowide m6100 | (b) profile x1.5 | 14.90 | 13.12 | 15.67 | 1.6 | 92.6 | 95.3 | 98.1 |
| 06_cont_servowide m6100 | (c) full servo x1 | 12.31 | 10.25 | 13.05 | 2.7 | 80.9 | 84.4 | 96.7 |
| 06_cont_servowide m6200 | (a) no servo | 12.92 | 10.50 | 13.80 | 3.5 | 91.2 | 96.9 | 98.8 |
| 06_cont_servowide m6200 | (b) profile x1.5 | 14.49 | 12.10 | 15.51 | 3.9 | 87.9 | 93.6 | 98.0 |
| 06_cont_servowide m6200 | (c) full servo x1 | 12.46 | 10.15 | 13.25 | 3.1 | 80.1 | 84.0 | 96.8 |
| 06_cont_servowide m6300 | (a) no servo | 12.76 | 10.19 | 13.70 | 4.3 | 90.6 | 97.3 | 98.8 |
| 06_cont_servowide m6300 | (b) profile x1.5 | 14.60 | 12.37 | 15.45 | 3.7 | 91.0 | 95.1 | 98.1 |
| 06_cont_servowide m6300 | (c) full servo x1 | 12.54 | 10.51 | 13.42 | 2.9 | 83.6 | 88.1 | 97.1 |
| 06_cont_servowide m6399 | (a) no servo | 12.74 | 10.25 | 13.64 | 4.1 | 91.0 | 97.1 | 98.7 |
| 06_cont_servowide m6399 | (b) profile x1.5 | 14.70 | 12.81 | 15.56 | 3.1 | 89.1 | 93.6 | 98.2 |
| 06_cont_servowide m6399 | (c) full servo x1 | 12.44 | 10.69 | 13.24 | 3.1 | 82.2 | 86.7 | 96.9 |

## hist3

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| hist3_calmt m4000 | LOWFRIC (-wristl1-hist3) | 15.45 | 9.66 | 16.49 | 5.5 | 78.9 | 84.6 | 89.0 |
| hist3_calmt m4100 | LOWFRIC (-wristl1-hist3) | 15.89 | 10.36 | 17.11 | 4.9 | 81.6 | 87.5 | 90.6 |
| hist3_calmt m4200 | LOWFRIC (-wristl1-hist3) | 15.48 | 8.74 | 16.76 | 6.1 | 79.7 | 86.7 | 90.6 |
| hist3_calmt m4300 | LOWFRIC (-wristl1-hist3) | 15.48 | 8.92 | 16.88 | 6.8 | 80.5 | 88.1 | 91.6 |
| hist3_calmt m4400 | LOWFRIC (-wristl1-hist3) | 15.82 | 10.36 | 16.88 | 4.5 | 80.7 | 86.9 | 92.4 |
| hist3_calmt m4500 | LOWFRIC (-wristl1-hist3) | 15.84 | 10.10 | 17.34 | 6.8 | 81.8 | 89.1 | 93.7 |
| hist3_calmt m4600 | LOWFRIC (-wristl1-hist3) | 15.84 | 10.19 | 17.30 | 7.0 | 82.8 | 90.6 | 93.7 |
| hist3_calmt m4700 | LOWFRIC (-wristl1-hist3) | 15.89 | 9.90 | 17.22 | 7.4 | 81.6 | 89.6 | 93.0 |
| hist3_calmt m4800 | LOWFRIC (-wristl1-hist3) | 15.97 | 10.14 | 16.92 | 5.3 | 83.4 | 89.8 | 93.4 |
| hist3_calmt m4900 | LOWFRIC (-wristl1-hist3) | 15.94 | 10.64 | 17.30 | 6.1 | 82.6 | 89.3 | 93.3 |
| hist3_calmt m5000 | LOWFRIC (-wristl1-hist3) | 16.20 | 10.80 | 17.51 | 4.5 | 84.0 | 88.9 | 94.6 |
| hist3_calmt m5100 | LOWFRIC (-wristl1-hist3) | 15.84 | 10.84 | 16.98 | 4.9 | 85.2 | 90.8 | 92.6 |
| hist3_calmt m5200 | LOWFRIC (-wristl1-hist3) | 15.95 | 10.40 | 17.06 | 5.5 | 83.8 | 89.8 | 94.7 |
| hist3_calmt m5300 | LOWFRIC (-wristl1-hist3) | 15.80 | 10.33 | 16.84 | 6.1 | 83.8 | 91.4 | 93.7 |
| hist3_calmt m5400 | LOWFRIC (-wristl1-hist3) | 16.24 | 11.01 | 17.33 | 5.1 | 85.5 | 91.2 | 94.7 |
| hist3_calmt m5498 | LOWFRIC (-wristl1-hist3) | 16.06 | 10.87 | 17.17 | 6.4 | 82.8 | 89.5 | 93.3 |
| hist3_calmt_servo m4000 | (a) no servo (-wristl1-hist3) | 15.86 | 10.71 | 16.82 | 4.7 | 81.6 | 86.5 | 90.0 |
| hist3_calmt_servo m4000 | (c) full servo x1 (-wristl1-hist3) | 8.30 | 2.04 | 9.09 | 20.5 | 26.4 | 44.7 | 64.8 |
| hist3_calmt_servo m4100 | (a) no servo (-wristl1-hist3) | 15.68 | 10.80 | 16.46 | 3.1 | 84.6 | 87.9 | 92.1 |
| hist3_calmt_servo m4100 | (c) full servo x1 (-wristl1-hist3) | 9.63 | 3.28 | 10.52 | 13.3 | 44.7 | 59.4 | 73.0 |
| hist3_calmt_servo m4200 | (a) no servo (-wristl1-hist3) | 15.30 | 9.54 | 16.34 | 5.1 | 82.6 | 88.5 | 92.3 |
| hist3_calmt_servo m4200 | (c) full servo x1 (-wristl1-hist3) | 10.58 | 3.42 | 11.90 | 12.9 | 53.7 | 65.8 | 79.3 |
| hist3_calmt_servo m4300 | (a) no servo (-wristl1-hist3) | 15.16 | 9.63 | 16.15 | 4.3 | 83.4 | 88.5 | 93.6 |
| hist3_calmt_servo m4300 | (c) full servo x1 (-wristl1-hist3) | 11.18 | 5.78 | 12.50 | 9.6 | 63.7 | 75.2 | 83.9 |
| hist3_calmt_servo m4400 | (a) no servo (-wristl1-hist3) | 14.90 | 9.52 | 16.00 | 5.1 | 85.7 | 92.4 | 93.1 |
| hist3_calmt_servo m4400 | (c) full servo x1 (-wristl1-hist3) | 11.31 | 5.24 | 12.69 | 9.8 | 62.5 | 73.8 | 83.8 |
| hist3_calmt_servo m4500 | (a) no servo (-wristl1-hist3) | 15.33 | 10.22 | 16.19 | 2.3 | 86.9 | 92.0 | 95.1 |
| hist3_calmt_servo m4500 | (c) full servo x1 (-wristl1-hist3) | 11.63 | 6.61 | 12.80 | 8.0 | 68.8 | 77.9 | 86.3 |
| hist3_calmt_servo m4600 | (a) no servo (-wristl1-hist3) | 14.94 | 9.54 | 15.91 | 4.1 | 83.2 | 89.3 | 93.6 |
| hist3_calmt_servo m4600 | (c) full servo x1 (-wristl1-hist3) | 11.41 | 5.25 | 12.83 | 9.8 | 63.9 | 74.8 | 84.5 |
| hist3_calmt_servo m4700 | (a) no servo (-wristl1-hist3) | 15.30 | 10.48 | 16.21 | 3.7 | 85.5 | 90.4 | 94.9 |
| hist3_calmt_servo m4700 | (c) full servo x1 (-wristl1-hist3) | 11.75 | 7.37 | 13.05 | 7.4 | 68.4 | 76.8 | 85.9 |
| hist3_calmt_servo m4800 | (a) no servo (-wristl1-hist3) | 14.86 | 10.03 | 15.68 | 4.5 | 82.2 | 88.3 | 95.0 |
| hist3_calmt_servo m4800 | (c) full servo x1 (-wristl1-hist3) | 11.55 | 7.34 | 12.74 | 7.6 | 64.6 | 74.2 | 85.7 |
| hist3_calmt_servo m4900 | (a) no servo (-wristl1-hist3) | 14.50 | 8.79 | 15.51 | 6.6 | 81.6 | 89.5 | 95.3 |
| hist3_calmt_servo m4900 | (c) full servo x1 (-wristl1-hist3) | 11.70 | 6.60 | 13.01 | 7.6 | 64.8 | 74.2 | 88.3 |
| hist3_calmt_servo m5000 | (a) no servo (-wristl1-hist3) | 15.05 | 9.99 | 16.22 | 5.1 | 85.0 | 91.2 | 94.7 |
| hist3_calmt_servo m5000 | (c) full servo x1 (-wristl1-hist3) | 11.14 | 3.32 | 12.84 | 11.9 | 63.7 | 76.4 | 85.0 |
| hist3_calmt_servo m5100 | (a) no servo (-wristl1-hist3) | 14.84 | 9.34 | 15.97 | 5.5 | 83.0 | 89.8 | 93.9 |
| hist3_calmt_servo m5100 | (c) full servo x1 (-wristl1-hist3) | 11.41 | 5.52 | 12.83 | 9.8 | 66.2 | 76.6 | 85.8 |
| hist3_calmt_servo m5200 | (a) no servo (-wristl1-hist3) | 14.80 | 9.02 | 15.92 | 6.8 | 86.1 | 93.6 | 94.6 |
| hist3_calmt_servo m5200 | (c) full servo x1 (-wristl1-hist3) | 11.87 | 7.65 | 13.11 | 7.4 | 68.2 | 77.3 | 87.2 |
| hist3_calmt_servo m5300 | (a) no servo (-wristl1-hist3) | 14.90 | 9.98 | 16.08 | 5.9 | 83.2 | 89.8 | 94.9 |
| hist3_calmt_servo m5300 | (c) full servo x1 (-wristl1-hist3) | 11.71 | 6.66 | 13.01 | 8.4 | 68.2 | 76.6 | 86.8 |
| hist3_calmt_servo m5400 | (a) no servo (-wristl1-hist3) | 14.75 | 9.80 | 15.54 | 4.7 | 84.6 | 90.8 | 95.3 |
| hist3_calmt_servo m5400 | (c) full servo x1 (-wristl1-hist3) | 11.94 | 8.03 | 13.12 | 6.2 | 69.5 | 77.1 | 87.8 |
| hist3_calmt_servo m5498 | (a) no servo (-wristl1-hist3) | 14.35 | 8.89 | 15.24 | 5.9 | 81.6 | 88.5 | 94.8 |
| hist3_calmt_servo m5498 | (c) full servo x1 (-wristl1-hist3) | 12.06 | 8.30 | 13.38 | 7.2 | 71.1 | 78.9 | 88.4 |
| hist3_calmt_cap16 m5500 | LOWFRIC (-wristl1-hist3) | 15.54 | 10.45 | 16.48 | 3.9 | 86.9 | 91.8 | 94.6 |
| hist3_calmt_cap16 m5600 | LOWFRIC (-wristl1-hist3) | 15.84 | 11.25 | 16.68 | 4.3 | 84.6 | 89.5 | 94.5 |
| hist3_calmt_cap16 m5700 | LOWFRIC (-wristl1-hist3) | 15.16 | 10.47 | 16.07 | 5.9 | 86.1 | 93.0 | 93.7 |
| hist3_calmt_cap16 m5800 | LOWFRIC (-wristl1-hist3) | 15.20 | 10.55 | 15.94 | 3.5 | 87.3 | 91.8 | 94.7 |
| hist3_calmt_cap16 m5900 | LOWFRIC (-wristl1-hist3) | 15.48 | 10.86 | 16.26 | 4.7 | 87.1 | 92.4 | 94.1 |
| hist3_calmt_cap16 m5999 | LOWFRIC (-wristl1-hist3) | 15.17 | 10.53 | 16.14 | 5.3 | 83.0 | 89.3 | 93.6 |

## ROA chain s42

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| roa_stage2_calm m4000 | LOWFRIC | 13.12 | 8.65 | 13.44 | 1.2 | 27.3 | 27.7 | 81.7 |
| roa_stage2_calm m4100 | LOWFRIC | 7.77 | 0.83 | 7.82 | 38.9 | 27.9 | 60.4 | 91.2 |
| roa_stage2_calm m4200 | LOWFRIC | 9.42 | 1.32 | 10.06 | 25.2 | 39.8 | 65.4 | 91.2 |
| roa_stage2_calm m4300 | LOWFRIC | 8.67 | 0.96 | 8.86 | 33.2 | 37.7 | 71.7 | 91.5 |
| roa_stage2_calm m4400 | LOWFRIC | 5.88 | 0.35 | 3.45 | 55.1 | 26.6 | 83.0 | 93.5 |

## LSTM chain

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| lstm_stage2_calm m4000 | LOWFRIC | 9.85 | 6.56 | 9.84 | 2.9 | 14.8 | 18.2 | 84.8 |
| lstm_stage2_calm m4100 | LOWFRIC | 8.97 | 3.48 | 9.10 | 17.4 | 26.4 | 42.2 | 93.1 |
| lstm_stage2_calm m4200 | LOWFRIC | 10.62 | 5.27 | 11.00 | 9.8 | 40.8 | 52.5 | 94.4 |
| lstm_stage2_calm m4300 | LOWFRIC | 10.09 | 3.64 | 10.53 | 15.0 | 46.3 | 66.6 | 96.8 |
| lstm_stage2_calm m4400 | LOWFRIC | 11.36 | 5.44 | 11.82 | 9.2 | 50.0 | 61.7 | 96.5 |
| lstm_stage2_calm m4500 | LOWFRIC | 11.57 | 4.78 | 12.12 | 10.5 | 52.9 | 63.1 | 96.6 |
| lstm_stage2_calm m4600 | LOWFRIC | 11.60 | 6.15 | 12.05 | 7.0 | 53.5 | 63.9 | 97.1 |
| lstm_stage2_calm m4700 | LOWFRIC | 12.68 | 8.24 | 13.20 | 4.1 | 66.0 | 72.1 | 97.9 |
| lstm_stage2_calm m4800 | LOWFRIC | 12.77 | 8.84 | 13.10 | 2.7 | 64.1 | 69.3 | 97.6 |
| lstm_stage2_calm m4900 | LOWFRIC | 12.52 | 7.94 | 13.07 | 4.5 | 64.8 | 72.3 | 98.0 |
| lstm_stage2_calm m5000 | LOWFRIC | 12.85 | 8.08 | 13.42 | 2.9 | 70.9 | 76.8 | 98.0 |

## base s4 chain

| checkpoint | eval | mean RPM | RPM p10 | RPM median | stall % | good-trial % | drop-free t>=2s % | in region % |
|---|---|---|---|---|---|---|---|---|
| 06s4_oneshot_calmt m4000 | LOWFRIC | 14.14 | 11.59 | 14.36 | 0.2 | 85.0 | 85.9 | 75.6 |
| 06s4_oneshot_calmt m4100 | LOWFRIC | 12.99 | 6.82 | 14.35 | 7.0 | 79.3 | 90.8 | 92.6 |
| 06s4_oneshot_calmt m4200 | LOWFRIC | 11.03 | 3.11 | 12.05 | 16.2 | 70.3 | 95.7 | 96.1 |
| 06s4_oneshot_calmt m4300 | LOWFRIC | 11.67 | 4.97 | 13.23 | 10.2 | 75.4 | 95.9 | 98.1 |
| 06s4_oneshot_calmt m4400 | LOWFRIC | 11.95 | 6.32 | 13.41 | 7.4 | 79.1 | 94.7 | 98.0 |
| 06s4_oneshot_calmt m4500 | LOWFRIC | 11.30 | 4.86 | 12.92 | 10.7 | 77.7 | 95.7 | 98.3 |
| 06s4_oneshot_calmt m4600 | LOWFRIC | 11.57 | 5.17 | 13.19 | 9.8 | 78.3 | 95.3 | 98.3 |
| 06s4_oneshot_calmt m4700 | LOWFRIC | 11.87 | 5.95 | 13.36 | 8.0 | 83.0 | 96.5 | 98.6 |
| 06s4_oneshot_calmt m4800 | LOWFRIC | 11.81 | 6.62 | 13.13 | 7.8 | 83.4 | 96.3 | 98.6 |
| 06s4_oneshot_calmt m4900 | LOWFRIC | 11.60 | 5.83 | 12.94 | 8.6 | 82.8 | 96.3 | 98.7 |
| 06s4_oneshot_calmt m5000 | LOWFRIC | 11.32 | 4.62 | 12.95 | 10.5 | 80.3 | 96.7 | 98.8 |
| 06s4_oneshot_calmt m5100 | LOWFRIC | 12.02 | 7.93 | 13.09 | 6.2 | 87.1 | 97.3 | 98.9 |
