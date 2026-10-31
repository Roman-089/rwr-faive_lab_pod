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
