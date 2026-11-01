# Deployable policies (2026-10-31, for the 10:00 Zurich deadline; updated 10:45Z)

User request (2026-10-30 22:30): "robust, deployable policies by 10 am". All candidates below are exported with
`scripts/export/export_history_policy.py` (TorchScript, normalization inside, bit-exact self-check against the training
policy) and evaluated in Isaac with the fixed protocol ([eval_protocol](eval_protocol.md): 64 envs × 60 s, +z at
1.5 rad/s, the task's DR + obs noise; "perturbed" adds 50–100 ms action delay and ±0.05 rad joint-encoder offset).
Numbers are for the exported checkpoint itself (nominal / perturbed); cube motion from `scripts/eval/cube_motion.py`;
MuJoCo and the course node in the loop are informational ([sim2sim](sim2sim.md)). All are +z only (the user's
2026-10-30 decision) except the fallback. The bundle's first version (W&B v0, 07:40Z) had #2, #3, #5, #6, #7 and
the fallback under their old numbers; v1 (10:45Z) adds #1 and #4 and this ranking.

## Ranking

| # | Policy (exported checkpoint) | Isaac RPM | drop-free % | drops / trial | cube tilt, face changes / min | wrist range / speed | MuJoCo RPM / drop-free | course node in the loop (MuJoCo), RPM |
|---|---|---|---|---|---|---|---|---|
| **1** | **Tilt + position ×3 + ADR** (`abl_B_onedir_droppen_upright_pos45_adr_r6k` model_5500, W&B s1jru0ly) | **14.2** / 16.0 | **98** / 72 | 0.02 / 0.36 | 11.6°, 2.8 | 0.37 rad / 0.32 rad/s | 7.1 / 92 | **7.3 ± 0.4** |
| **2** | **Tilt penalty + ADR** (`abl_B_onedir_droppen_upright_adr_r6k` model_6998, W&B 6rsbkqqm) | 11.6 / 13.5 | 91 / **83** | 0.12 / 0.25 | **7.4°, 2.5** | **0.28 / 0.08** | 7.1 / 92 (5500) | 6.2 ± 1.6 |
| 3 | Position ×3 + ADR (`abl_B_onedir_droppen_pos45_adr_from5e-3_r6k` model_7000, W&B on6yv8hs) | 12.3 / 15.8 | 95 / **84** | 0.05 / 0.19 | 16.5°, 3.5 | 0.34 / 0.52 | 5.7 / 92 | – |
| 4 | Tilt + wrist penalty, no ADR (`abl_B_onedir_droppen_upright_wristpen_ent1e-4` model_4999, W&B 1xxwpij9) | 13.4 / 11.8 | 91 / 63 | 0.11 / 0.59 | 7.0°, 2.3 | 0.33 / 0.10 | 7.4 / 89 | – |
| 5 | Tilt + position ×3, no ADR (`abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k` model_3999, W&B b0r506gz) | 14.5 / 11.5 | 92 / 45 | 0.09 / 1.03 | 10.4°, 2.3 | 0.31 / 0.29 | 7.6 / 86 | – |
| 6 | Position ×3, default entropy, no ADR (`abl_B_onedir_droppen_pos45_ent1e-4_r6k` model_3999, W&B kfuq4jwb) | 15.5 / 14.4 | 92 / 42 | 0.08 / 1.38 | 16.8°, 3.8 | 0.46 / 0.43 | 4.6 / 88 | – |
| 7 | Tilt penalty, no ADR (`abl_B_onedir_droppen_upright_ent1e-4_r6k` model_4000, W&B 5oklh054) | 11.2 / 11.3 | 94 / 63 | 0.06 / 0.73 | 7.4°, 2.7 | 0.28 / 0.09 | 7.2 / 84 | – |
| fallback | Transformer student, **bidirectional** (`student_pt_rotlin` model_1999, W&B p3dxn5pm) | 10.9 / 9.8 | 86 / 56 | 0.17 / 0.64 | 6.7°, 3.8 | 0.48 / 0.23 | 4.5 / 67 | – |

Cube motion is from the nominal rollout of the run's last checkpoint (#1: 5998, #2: 6998, #3: 7998, #4: 4999,
#5 / #6: 3999, #7: 4999). The in-the-loop column: 16 × 60 s trials of the unchanged course node against the MuJoCo
hand ([sim2sim](sim2sim.md#software-in-the-loop-the-course-node-against-mujoco-2026-10-31)); both #1 and #2 held the
cube for the full minute in every trial (the only drops are 2 spawn artifacts in the first 0.6 s, identical for every
policy and command).

**Recommendation.** **#1** is the fastest robust policy (14 RPM in Isaac, 2.6 more than #2 at a similar
delay / offset tolerance) and the most consistent in the loop; **#2** is the conservative choice (flattest cube,
almost still wrist, highest drop-free rate under delay / offset). Suggested first hardware session: #2 to check that
everything works, then #1. The ADR policies (#1–#3) drop 3–5× less under delay / offset than their non-ADR versions
(#5, #7) at the same nominal numbers, so prefer them over #4–#7. Caveats: #2 / #7 build on seed 0 of the tilt penalty, a
good draw (seed 1: 9.6 RPM / 74 %); seeds 1 and 2 of #5 / #1's base config are queued. The +z speed counts rotation
about world z only; all ranked policies keep the cube flat, so it is not inflated by tumbling.

## Deploy package (for the hand)

W&B artifact **`rwr-fast/orca_v2_repose/orca_v2_zrot_deploy`** (v0, aliases `latest`, `2026-10-31`): the three ADR
policies (#1–#3) as `<n>_<name>/policy.pt` + `policy.json`, a README for the person at the hand, `run_policy.sh <n>`
(runs the unchanged course node with `goal_ang_vel:=1.5`, flip period 0; extra `-p` args pass through) and
`manifest.json` (checkpoint, W&B run, Isaac / MuJoCo / in-the-loop metrics, sha256 per policy). Download:
`wandb artifact get rwr-fast/orca_v2_repose/orca_v2_zrot_deploy:latest`.

## Real-hand fine-tunes (2026-10-31 afternoon)

W&B **`rwr-fast/orca_v2_repose/orca_v2_zrot_hand_ft`**: 07 fine-tuned on the real hand's friction (TPU palm 0.3–0.5,
silicone pads 0.55–0.9, measured), a wrist penalty and delay / encoder-offset DR, after 17 + 4 hand trials showed that
wrist motion (palm tipping on a slippery palm) is what drops the cube. v8 (latest): A = `07ft_s0_m4600` (recommended: 11.3 RPM, 97 % drop-free, 0.05 drops, wrist 0.08 rad/s at the measured
friction; 07: 11.1 / 88 % / 0.20 / 0.14), B = `07rf2_s0_m5200` (also trained with the PLA skeleton's low friction: 11.1
RPM, 98 %, 0.02 drops), plus `07ft_s1_m4300`, `07ft_s1_m4150`, `07ft_s0_m4150`. Same interface and launch as #7. Details:
[sim2real](sim2real.md#real-hand-servo-model-as-dr-2026-10-31).

**512-env re-eval (17:40Z): A and B are not better than the other fine-tunes.** At the measured friction with 512 envs (drop-free SE ±1.3 %) all eight candidates sit at 89–92 % drop-free, 11.2–11.8 RPM, wrist 0.06–0.08 rad/s: A 90.0 % (11.41 RPM), B 89.3 % (11.15), 07ft_s0 m5200 91.4 %, **07ft_s1 m4900 92.2 % (11.39, nominal leader, n.s.)**, region200c m4800 90.0 %, region200 m5000 90.6 %, roll10 m4600 91.6 %, m5000 90.0 %. Their 64-env scores (A 96.9 %, B 98.4 %) were the best of many noisy evals (SE ±3.7 %). The ranking A > B in `deploy_policies/` is therefore a coin flip in sim: let the hand decide. **Ship rule from now on:** the candidate must beat the current best on the 512-env measured-friction eval by more than 2 SE (~2.5 points drop-free or a clear RPM gain at equal drop-free), plus the own-task eval and wrist ≤ ~0.1 rad/s. 64-env evals only pre-select.

**Region variant shipped (2026-11-01 02:35Z): `07region_s1_m5500`**, W&B `orca_v2_zrot_hand_ft:v9` (latest, all six folders), `deploy_policies/` rank 3 (sha256 `9ee5647b…`). A fine-tuned 900 iterations (seed 1, DTU) on `-realfric-region200c-v0`: the user's circle, centre (−11.6, 13.6) mm, r 27.5 mm, weight 200 with exponential fall-off, and the tracking reward pulled to the circle centre. 512 envs at the measured friction: 11.05 RPM, 90.2 % drop-free (95.3 % after 2 s), wrist 0.06 rad/s, axis in the circle **97.0 %** (A 93.2 %, B 91.5 %); own task 14.16 RPM / 93.8 % (64 envs). It is the only region checkpoint that also passes its own task (the centred-spawn variant: 70–80 %). Same interface as A/B. Ranks 1/2 got their 512-env numbers in SPEC.md / README (6d62d01).

**New rank 1 (2026-11-01 04:30Z): `06calm_s0_m7000`**, W&B `orca_v2_zrot_hand_ft:v10`, `deploy_policies/` rank 1 (1290aca, notes a537fc0; sha256 `ae005253…`). The fast 06 line made calm in four fine-tune stages (06_pos45 m3999 → 06_wp_rf: real friction + wrist penalty → 06_calm: tilt −400 / wrist −8 → 06_calm2: wrist −16 → 06_calm3_l1: + L1 wrist −40, m7000). 512 envs at the measured friction: **12.93 RPM** vs 11.41 for 07ft_s0_m4600 at the same drop-free (90.0 %, 0.109 drops per trial), 98.2 % after 2 s (96.5), axis **98.8 % in the user's circle** (93.2), tilt 7.2° (6.1), wrist 0.113 rad/s all envs (0.079); own task 16.0 RPM. Calmer than the original 07 (~0.18 all-env equivalent), which ran the full 60 s on the hand. New lineage: its ancestor 06_pos45 swung the wrist on the hand, so watch it on the first run (rank 2 is the calm fallback).

**New rank 1 (2026-11-01 05:20Z): `06calmt_s0_m6300`**, W&B `orca_v2_zrot_hand_ft:v11`, `deploy_policies/` rank 1 (8ab2aa8, notes 2c0b14b; sha256 `9af61ac0…`); `06calm_s0_m7000` → rank 2. 06_calm m5800 fine-tuned with tilt −800 and the L1 wrist term (−40) (`06_calm3_l1_tilt800`, m6300). 512 envs at the measured friction: 12.46 RPM, 89.5 % drop-free (0.121 drops per trial), 98.4 % after 2 s, axis 99.0 % in the user's circle, **tilt 4.8°, wrist 0.086 rad/s all envs**, as calm as the 07 line (A: 6.1°, 0.079). Against m7000: same drop-free / region, −24 % wrist, −2.4° tilt, −0.5 RPM. The hand dropped the cube through wrist motion, so this one goes first.

Replicated (06:40Z): the same recipe with seed 1, m6300 at 512 envs: 12.57 RPM / 88.1 % / 98.6 % after 2 s / 99.0 % in region / tilt 4.9° / wrist 0.083 (rank 1: 12.46 / 89.5 / 98.4 / 99.0 / 4.8° / 0.086), and it tracks seed 42 checkpoint by checkpoint (tilt within 0.6°, wrist within 0.012). The tilt −800 recipe is seed-robust; the earlier calm3 recipe (rank 2) ran 1–2° tiltier with seed 1. Seed 1's m6300 is on the pod (26e4e46) as a spare.

Rank 1 is mount-robust as it is: on its own task with the hand base rolled ±10° (instead of ±3°), 512 envs: 15.10 RPM / 83.0 % / 96.7 % after 2 s, vs 15.54 / 85.0 / 97.1 at ±3°: only ~2 points of drop-free lost. (The 07 line lost ~10 points on the ±10° mount10 task.) A ±10° roll fine-tune of rank 1 (`06_calmt_roll10`) is 1–2 points ahead on the tilted mount after 100 iterations, inside the noise.

Third seed (DTU, seed 2, 512 envs, all-env wrist): m6500 12.60 RPM / 88.3 % / 98.2 % after 2 s / 99.1 % in region / tilt 4.6° / wrist 0.08; m6799 13.05 / 89.8 / 98.2 / 99.1 / 4.6° / 0.08 (seed 42 m6799: 12.98 / 87.7 / 98.6 / 99.1 / 4.7° / 0.084). Three seeds of the rank-1 recipe agree within noise.

Rank 1 on the PLA-skeleton friction (`-realfric2-v0`, its own DR: bare PLA links 0.0–0.2, cube 0.2–0.3; 512 envs): 15.69 RPM / 84.0 % / 97.3 % after 2 s / 98.3 % in region / tilt 4.8° / wrist 0.085, vs B (trained on it) 14.20 / 85.5 / 95.3 / 90.2 / 6.0° / 0.078. The bare-PLA contact does not break rank 1, and B has no advantage there; B's case for rank 4 rests only on the hand.

**Keep `goal_ang_vel` at 1.5 on the hand.** Rank 1 fine-tuned 100 iterations toward a 2.0 rad/s command (`06_calmt_cmd2`, m6400, 512 envs) turns slower at 2.0 than at 1.5: 8.60 RPM / 84.6 % / 93.8 % after 2 s, vs 12.08 / 88.5 / 98.8. A command outside the training range is not a speed knob. Deploy-rwr was told.

**New rank 1 (2026-11-01 08:55Z): `06oneshot_s0_m5400`**, W&B `orca_v2_zrot_hand_ft:v12`, `deploy_policies/` rank 1 (2813933, notes f292ebb; sha256 `f01377bb…`); the rest shift down one. The raw 06 base (pos45 m3999) calmed in **one** fine-tune stage of 1400 iterations with the final recipe (`-realfric-wristl1-v0`, tracking 45, tilt −800, L2 wrist −16, L1 wrist −40). 512 envs at the measured friction: **13.07 RPM** (previous rank 1 12.46) at the same drop-free (89.5 %), 98.8 % after 2 s, 98.9 % in the user's circle, tilt 4.1°, wrist 0.074 rad/s all envs: the calmest so far. Own task 15.69 RPM / 85.2 %.

**New rank 1 (2026-11-01 12:15Z): `06servo_s0_m5800`**, W&B `orca_v2_zrot_hand_ft:v13`, `deploy_policies/` rank 1 (e471b63, notes 1bdb85e; sha256 `b9496214…`); 06oneshot_s0_m5400 → rank 2. 06oneshot_s0_m5400 fine-tuned 400 iterations on `-realfric-wristl1-servofast-v0`, with the servo model as DR (0–67 ms delay, play, acceleration limit 1.5–3× the fitted value). 512 envs at the measured friction: no servo 12.68 RPM / 88.9 % / 99.2 % after 2 s / 99.0 % in region / wrist 0.073; acceleration limit 1.5× 14.56 / 77.5 / 95.7; full fitted servo model 12.23 / 45.1 / 84.4 (rank 2: 13.07 / 89.5 / 98.8, 13.19 / 75.0 / 90.8, 9.95 / 40.8 / 76.8). Every checkpoint m5500–m5999 beats rank 2 under the servo model (+4–6 and +6–10 points after 2 s), so the gain is stable. The cost is 0.4 RPM with ideal servos.

**Per-env speed spread (user, 2026-11-01 12:50Z).** The mean RPM hides envs that nearly stall: in the rank-1 rollout the 64 envs range from ~1 to ~16 RPM (std 4). One env is one draw of the randomised friction / mass, like one hand trial, so the spread is the spread of hand outcomes. `cube_motion.py --spread` (718fbce) reports per-env RPM p10 / median, **stall rate** (envs < 5 RPM) and **good-trial rate** (no drop after 2 s and ≥ 8 RPM). 64-env first look: rank 1 06servo p10 4.36 / median 13.68 / stall 12.5 % / good 81.2 %; rank 2 06oneshot 8.34 / 14.45 / 4.7 % / 85.9 %; A 7.64 / 11.85 / 4.7 % / 84.4 %; B 7.24 / 12.07 / 6.2 % / 85.9 %. The servo-hardened rank 1 may stall more with ideal servos (1.5 SE at 64 envs); the 512-env recompute decides. Good-trial rate becomes the main ship metric once the 512 numbers are in.

Servo-model stress test (512 envs, measured friction + `--servo`: 0–50 ms delay, nominal acceleration limit, play; the model is harsher than the hand, it is a relative check, not a prediction): rank 1 9.95 RPM / 40.8 % / 76.8 % after 2 s / 96.3 % in region; rank 2 (06calmt) 9.59 / 39.1 / 78.5 / 96.4; A 11.09 / 32.0 / 56.4 / 73.1. The calm 06 line loses more speed but holds the cube much better than the 07 line under servo dynamics: the faster gait is not the risk.
Per part (rank 1 | A, 512 envs; drop-free / after 2 s / in region): delay only 89.8 / 98.2 / 98.9 | 88.5 / 93.9 / 93.3 (both ~2.5 RPM *faster*); play only 89.8 / 99.2 / 98.6 | 92.2 / 98.2 / 88.0; **acceleration profile only 52.3 / 78.5 / 97.0 | 38.9 / 54.5 / 80.2**, rank 1's wrist doubling (0.074 → 0.164). The nominal per-joint acceleration limit does nearly all the damage. It was fitted from hand replays (kinematics only) and ranked 07, the hand's best, last in closed loop, so the servo fine-tunes were dropped. No profile-DR training until the hand tests of rank 1. `eval_policy.py --servo_amax_scale` (33fd4ad) scans how much faster servos would have to be.

Acceleration-limit scan (`--servo --servo_parts profile --servo_amax_scale K`, 512 envs; rank 1 | A, RPM / drop-free / after 2 s / in region):

| amax | rank 1 06oneshot_s0_m5400 | A 07ft_s0_m4600 |
|---|---|---|
| no servo | 13.07 / 89.5 / 98.8 / 98.9 | 11.41 / 90.0 / 96.5 / 93.2 |
| ×1 (fitted) | 10.29 / 52.3 / 78.5 / 97.0 | 11.80 / 38.9 / 54.5 / 80.2 |
| ×1.5 | 13.19 / 75.0 / 90.8 / 98.3 | 13.42 / 55.7 / 68.2 / 88.5 |
| ×2 | 14.75 / 81.4 / 94.7 / 98.4 | 13.96 / 70.3 / 79.3 / 91.4 |
| ×3 | 15.62 / 84.6 / 96.9 / 98.7 | 14.23 / 83.4 / 92.4 / 93.3 |

Rank 1 recovers at ~2× the fitted limit, A only at ~3×; at every scale rank 1 is 12–24 points ahead after 2 s. What to read from the hand: if rank 1 runs close to its sim numbers and A degrades, the real servos sit around 1.5–2× the fitted limit; if both run about equally, the headroom is ≥ 3× (or the profile model is off, as with 07 before). In both cases rank 1 is the better bet.

## All exported policies

W&B artifact **`rwr-fast/orca_v2_repose/orca_v2_all_policies`** (v0, 12:07Z): every proprioceptive policy exported
so far, 19 checkpoints of 18 runs (+z and bidirectional, incl. the tumbling ones for comparison and both transformer
students), `manifest.json` with direction, `goal_ang_vel`, history / `reset()` line, Isaac metrics and sha256.

## Files

On DTU under `~/rwr_faive_lab/logs/rsl_rl/orca_v2_repose/<run dir>/exported/` (`policy_history.pt` + `.json`):

| # | run dir |
|---|---|
| 1 | `2026-10-31_06-26-57_abl_B_onedir_droppen_upright_pos45_adr_r6k` |
| 2 | `2026-10-31_02-41-27_abl_B_onedir_droppen_upright_adr_r6k` (model_5500 alternative in `exported_model5500/`) |
| 3 | `2026-10-30_22-18-44_abl_B_onedir_droppen_pos45_adr_from5e-3_r6k` |
| 4 | `2026-10-30_23-02-12_abl_B_onedir_droppen_upright_wristpen_ent1e-4` (trained on DTU) |
| 5 | `2026-10-31_01-07-40_abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k` |
| 6 | `2026-10-31_00-49-37_abl_B_onedir_droppen_pos45_ent1e-4_r6k` |
| 7 | `2026-10-30_18-43-26_abl_B_onedir_droppen_upright_ent1e-4_r6k` |
| fallback | `2026-10-30_01-55-31_student_pt_rotlin` |

Also bundled as the W&B artifact `rwr-fast/orca_v2_repose/deployable_policies_2026-10-31` (type `model`, alias
`latest`; one folder per candidate with `policy_history.pt`, `policy_history.json` and this file). All but #4 trained on
the pod; their run dirs on DTU were rebuilt from W&B / the pod's pushed checkpoints, with `params/` copied from a DTU
run of the same network and observation layout (`params/SOURCE.txt`; the run's own config in
`params/wandb_config_<id>.json`).

## How to run them

**With the course node** (`rwr_system/experiments/rl_inference/policy_inference_cube_spin_v2.py`, inspected
2026-10-31 and run unchanged in the loop, [export](export.md#the-real-policy-node-rwr_system-inspected-2026-10-31)):
#1–#7 need **no node change**, only the right parameters:

```bash
ros2 run experiments policy_inference_cube_spin_v2.py --ros-args \
    -p policy_ckpt_path:=<run dir>/exported/policy_history.pt -p goal_ang_vel:=1.5   # flip period stays 0
```

**`goal_ang_vel:=1.5` is required.** The node's default is 1.0; the +z policies were trained on exactly +1.5, and with
1.0 #2 holds the cube but does not turn it (0.1 RPM in the loop vs 6.2 at 1.5). The fallback student (10-step history)
needs one line in the node, `self.policy.reset()` right after the warm-up call `self.policy(torch.zeros(1, OBS_DIM))`,
and one node start per trial; it takes the default `goal_ang_vel` (trained on |w| 0.9–1.5, either sign).

- **Interface = the default repo's `policy.pt`:** 74-dim observation per 50 ms step, float32, `[1, 74]` or `[74]`:
  `joint_pos` (17, limit-normalized to [−1, 1]), `joint_vel` (17, × 0.2), `goal_command` (6: (0, 0, −0.05) and the
  angular velocity), `last_action` (17), `last_joint_pos_cmd` (17). No noise. Output: 17 actions in [−1, 1] → joint
  targets through EMA α 0.2, rescaled to the joint limits. The normalizer is inside the file. In the loop, the node's
  observation and commands match our training-side code to ≤ 6e-7 over 1199 steps.
- **#1–#7 are memoryless MLPs** (history length 0): the node's warm-up call does not matter.
- Robustness expectations from sim: ADR trained #1–#3 under 0–125 ms action delay and ±0.08 rad encoder offset; they
  turn *faster* under 50–100 ms delay. The real driver's timing (100 Hz state, 10 ms command latency) costs nothing in
  the loop (6.0 vs 6.2 RPM for #2). None of the policies has seen servo dynamics, backlash or tendon effects beyond the
  env's gain / friction DR (actuator net not used yet, [actuator_net](actuator_net.md)).
- On the hand: 60 s trials, `-p csv_log_path:=...` to log observations and commands; joint targets vs measured
  positions give the real delay and offsets (they set the ADR ranges of the next round).

## Coming next

- ADR fine-tunes of #4 (`upright_wristpen_adr`, DTU A100) and #6 (`pos45_adr_ent1e-4`, DTU A40), queued.
- Teacher `teacher_onedir_droppen_upright_pos45` (pod) → transformer student (2000 it; needs the node's `reset()` line).
- Seeds 1 and 2 of #5, tilt + position ×3 + smoothness ×10 / + wrist penalty, ADR from scratch (pod), tilt dose −100.
