# Deployable policies (2026-10-31, for the 10:00 Zurich deadline)

User request (2026-10-30 22:30): "robust, deployable policies by 10 am". All candidates below are exported with
`scripts/export/export_history_policy.py` (TorchScript, normalization inside, bit-exact self-check against the training
policy) and evaluated in Isaac with the fixed protocol ([eval_protocol](eval_protocol.md): 64 envs × 60 s, +z at
1.5 rad/s, the task's DR + obs noise; "perturbed" adds 50–100 ms action delay and ±0.05 rad joint-encoder offset).
Numbers are for the exported checkpoint itself (nominal / perturbed); cube motion from `scripts/eval/cube_motion.py`;
MuJoCo is informational ([sim2sim](sim2sim.md)). All are +z only (the user's 2026-10-30 decision) except the fallback.

## Ranking

| # | Policy (exported checkpoint) | RPM | drop-free % | drops / trial | cube tilt mean, face changes / min | wrist range / speed | MuJoCo RPM / drop-free |
|---|---|---|---|---|---|---|---|
| **1** | **Tilt penalty + ADR** (`abl_B_onedir_droppen_upright_adr_r6k` model_6998, W&B 6rsbkqqm) | 11.6 / 13.5 | **91 / 83** | 0.12 / 0.25 | **7.4°, 2.5** | **0.28 rad / 0.08 rad/s** | 7.1 / 92 (model_5500) |
| 2 | Position ×3 + ADR (`abl_B_onedir_droppen_pos45_adr_from5e-3_r6k` model_7000, W&B on6yv8hs) | 12.3 / 15.8 | **95 / 84** | 0.05 / 0.19 | 16.5°, 3.5 | 0.34 / 0.52 | 5.7 / 92 |
| 3a | Tilt + wrist penalty (`abl_B_onedir_droppen_upright_wristpen_ent1e-4` model_4999, W&B 1xxwpij9; added 08:50Z) | 13.4 / 11.8 | 91 / 63 | 0.11 / 0.59 | **7.0°, 2.3** | 0.33 / 0.10 | 7.4 / 89 |
| 3 | Tilt + position ×3 (`abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k` model_3999, W&B b0r506gz) | **14.5** / 11.5 | 92 / 45 | 0.09 / 1.03 | 10.4°, 2.3 | 0.31 / 0.29 | **7.6** / 86 |
| 4 | Position ×3, default entropy (`abl_B_onedir_droppen_pos45_ent1e-4_r6k` model_3999, W&B kfuq4jwb) | **15.5** / 14.4 | 92 / 42 | 0.08 / 1.38 | 16.8°, 3.8 | 0.46 / 0.43 | 4.6 / 88 |
| 5 | Tilt penalty, no ADR (`abl_B_onedir_droppen_upright_ent1e-4_r6k` model_4000, W&B 5oklh054) | 11.2 / 11.3 | 94 / 63 | 0.06 / 0.73 | 7.4°, 2.7 | 0.28 / 0.09 | 7.2 / 84 |
| fallback | Transformer student, **bidirectional** (`student_pt_rotlin` model_1999, W&B p3dxn5pm) | 10.9 / 9.8 | 86 / 56 | 0.17 / 0.64 | 6.7°, 3.8 | 0.48 / 0.23 | 4.5 / 67 |

Cube motion is from the nominal rollout of the run's last checkpoint (#1: 6998, #2: 7998, #3 / #4: 3999, #5: 4999),
which for #2 and #5 is not the exported one.

**Recommendation: #1** for the first hardware runs: the cube turns flat about z (the demo task), the wrist barely
moves, and it is the most tolerant of delay and encoder offset among the flat policies (drops under perturbation
−66 % vs #5, same nominal speed). **#2** is equally robust but tilts the cube more and swings the wrist ~6× faster.
**#3a** (added 08:50Z, after the bundle) is the most robust policy without ADR (63 % under perturbation) at 13.4 RPM
with the cube as flat as #1; its ADR fine-tune is queued. **#3 / #4** are the faster options (14–15 RPM) for when the hand turns out to tolerate the sim-to-real gap; their ADR
versions are running (`upright_pos45_adr` on the pod, eval ~09:40Z; `pos45_adr_ent1e-4` queued on DTU A40) and will
replace them here. Caveats: #1 / #5 come from seed 0 of the tilt penalty, a good draw (seed 1: 9.6 RPM / 74 %); the
seed spread of #3 is being measured (seeds 1 and 2 queued). The +z speed numbers count rotation about world z only;
all ranked policies keep the cube flat, so they are not inflated by tumbling. In MuJoCo (informational) the tilt-penalty
policies keep the most of their speed (#1 67 %, #3 52 %, #5 64 %) and #4 the least (30 %): of the two fast options, #3
is the safer bet.

## Files

On DTU under `~/rwr_faive_lab/logs/rsl_rl/orca_v2_repose/<run dir>/exported/` (`policy_history.pt` + `.json`):

| # | run dir |
|---|---|
| 1 | `2026-10-31_02-41-27_abl_B_onedir_droppen_upright_adr_r6k` (model_5500 alternative in `exported_model5500/`) |
| 2 | `2026-10-30_22-18-44_abl_B_onedir_droppen_pos45_adr_from5e-3_r6k` |
| 3a | `2026-10-30_23-02-12_abl_B_onedir_droppen_upright_wristpen_ent1e-4` (trained on DTU) |
| 3 | `2026-10-31_01-07-40_abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k` |
| 4 | `2026-10-31_00-49-37_abl_B_onedir_droppen_pos45_ent1e-4_r6k` |
| 5 | `2026-10-30_18-43-26_abl_B_onedir_droppen_upright_ent1e-4_r6k` |
| fallback | `2026-10-30_01-55-31_student_pt_rotlin` |

Also bundled as the W&B artifact `rwr-fast/orca_v2_repose/deployable_policies_2026-10-31` (type `model`; one folder
per candidate with `policy_history.pt`, `policy_history.json` and this file). Runs #1–#4 trained on the pod; their
run dirs on DTU were rebuilt from W&B / the pod's pushed checkpoints, with `params/` copied from a DTU run of the same
network and observation layout (`params/SOURCE.txt`; the run's own config in `params/wandb_config_<id>.json`).

## How to run them (deploy contract, [export](export.md#deploy-contract))

**With the course node** (`rwr_system/experiments/rl_inference/policy_inference_cube_spin_v2.py`, inspected
2026-10-31, [export](export.md#the-real-policy-node-rwr_system-inspected-2026-10-31)): #1–#5 and #3a need **no node
change**, only the right parameters:

```bash
ros2 run experiments policy_inference_cube_spin_v2.py --ros-args \
    -p policy_ckpt_path:=<run dir>/exported/policy_history.pt -p goal_ang_vel:=1.5   # +z, as trained; flip period stays 0
```

(the node's default `goal_ang_vel` 1.0 is an input the +z policies never saw). The fallback student (10-step history)
needs one line in the node, `self.policy.reset()` right after the warm-up call `self.policy(torch.zeros(1, OBS_DIM))`,
and one node start per trial; it takes the default `goal_ang_vel` (trained on |w| 0.9–1.5, either sign).


- **Interface = the default repo's `policy.pt`:** 74-dim observation per 50 ms step, float32, `[1, 74]` or `[74]`, in
  this order: `joint_pos` (17, limit-normalized to [−1, 1]), `joint_vel` (17, × 0.2), `goal_command` (6: target cube
  position in the env frame, then target angular velocity), `last_action` (17), `last_joint_pos_cmd` (17). No noise.
  Output: 17 actions in [−1, 1] → joint targets through EMA α 0.2, rescaled to the joint limits (as for `policy.pt`).
  The normalizer is inside the file.
- **Command +z:** angular-velocity part (0, 0, +1.5) rad/s. The +z policies were trained on +z only (the magnitude is
  not tracked: they turn as fast as they can). The fallback takes ±z.
- **#1–#5 are memoryless MLPs** (history length 0): `policy_history.pt` can replace `policy.pt` directly; `reset()` is
  a no-op. The **fallback has a 10-step history**: call `policy.reset()` at every episode start and `forward` exactly
  once per 50 ms tick.
- Robustness expectations from sim: ADR trained #1 / #2 under 0–125 ms action delay and ±0.08 rad encoder offset; both
  turn *faster* under 50–100 ms delay (they were trained mostly with delay). None of the policies has seen servo
  dynamics, backlash or tendon effects beyond the env's gain / friction DR (actuator net not used yet,
  [actuator_net](actuator_net.md)).
- First hardware test suggestion: #1, cube placed as in sim, 60 s trials; log joint targets vs measured positions to
  estimate the real delay and offsets (they set the ADR ranges for the next round).

## Coming after the deadline

- `upright_pos45_adr` (pod, eval ~09:40Z): ADR on #3 → a fast (14 RPM) flat robust policy, the likely new #1.
- `pos45_adr_ent1e-4` (DTU A40, queued): ADR on #4.
- Teacher `teacher_onedir_droppen_upright_pos45` (pod, ~09:20Z) → transformer student (2000 it, afternoon).
- `upright_wristpen_adr` (DTU A100, queued): ADR on #3a.
- Seeds 1 and 2 of #3, tilt + position ×3 + smoothness ×10, tilt-penalty dose (−100 / −500), ADR from scratch.
