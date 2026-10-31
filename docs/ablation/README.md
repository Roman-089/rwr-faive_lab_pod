# Ablation study: RWR26 Task 1, Subtask 1

Proprioceptive in-hand z-rotation of a 45 mm cube with the ORCA v2 hand in Isaac Lab
(task `orca-v2-inhand-repose-v0`, branch `ablation-study`). Demo hardware is a plain ORCA hand without joint sensors,
so the deployed policy sees motor-encoder proprio only (74-dim obs at 20 Hz).

## Phases

- **Phase 1 (now): architecture + reward study** (decided 2026-10-28: find the best architecture and reward first, then
  iterate on sim2real). All runs on the **baseline env (A)**, compared by a fixed eval: MLP (A) · asymmetric critic (B) ·
  critic + DR params (C) · history (hist3 / hist10) · LSTM actor-critic · RMA · privileged teacher → distilled students
  (MLP-h10, LSTM, transformer + recon) · reward variants · entropy. Next candidates, on the winner of the reward wave
  (B, `-rotlin` or `-rotsoft`, decided by eval): `alpha05`, `relact`, `big`, student fine-tune.
- **Phase 2: sim2real** (parked). Sim2real ablations change the training distribution and can only be judged on
  hardware. The first Phase 2 run is a **fixed, enforced action delay** matched in sim and on deploy
  ([sim2real](sim2real.md#fixed-action-delay-sim--deploy), `DelayedEMAJointPositionToLimitsActionCfg`).

## Key results so far (2026-10-29 10:00)

- **Reward is the largest lever:** a rotation reward linear in the speed about the commanded axis (`rotlin`, as in
  HORA / LEAP) gives **10.33 RPM (two seeds: 10.41 / 10.25)** vs 5.45 for the same setup with the old reward (+90 %).
  A drop penalty adds +2.0 RPM on B (two seeds: 7.47 vs 5.45). Costs: 84 % drop-free (B 88 %), weaker under the perturbed eval (44 vs ~55 % drop-free), 2× the off-axis wobble.
- **Asymmetric critic** (privileged critic, deployable actor): A 1.37 → B 5.45 RPM (two seeds). Extra critic inputs (C:
  cube velocity, DR parameters) add nothing (5.34, two seeds). Seed spread is ~±0.7 RPM.
- **Entropy** 1e-4 → 5e-3 → 1e-2: 5.45 → 6.54 (two seeds) → 7.60 RPM (one seed); drops rise at 1e-2. It also makes the
  privileged teacher learn ~2.5× faster (8 RPM at 4k iterations).
- **History / memory** (hist3, hist10, LSTM, RMA): no speed gain, mainly robustness. The **transformer student**
  (`student_pt`) keeps 93 % of the teacher's speed (6.89 RPM) with the best drop-free rate of any deployable policy
  (95 %); its export with the history inside `policy_history.pt` is verified in sim ([export](export.md)).
- **Failed / ruled out:** `rotsoft` and an unclipped speed reward (both reward hacks), relative joint targets without
  grasp-cache resets, C. The policies ignore the commanded speed, so the demo target is one direction at max speed.
- Domain randomization: the no-DR policy reaches 10.55 RPM but collapses to 3.3 with DR; a DR-trained policy loses only
  ~15 % when tested without DR, so most of the DR cost is a more conservative gait.

- **Sim2sim in MuJoCo** ([sim2sim](sim2sim.md), CPU, USD-derived hand model, Isaac-identical interface, 47 tests):
  no interface bug; every policy grips and turns the cube the commanded way, but at **48–73 % of its Isaac speed for
  the MLPs** (rotlin 5.39, ent5e-3 5.38, droppen 4.39 RPM, 32 trials) and **~40 % for `student_pt`** (2.79 RPM, drops
  mid-trial). rotlin's Isaac lead disappears in MuJoCo (MuJoCo's cube torsional friction alone can halve RPM).
  **Decision rule (2026-10-29): MuJoCo results are informational only; design decisions for new ablations are based
  solely on the Isaac eval.** The real hand decides deployability.

**Deployable?** Every `abl_*` run, `arch_asym_*`, `arch_rma_s2` and every `student_*` is deployable: the actor sees only
the 74 proprioceptive observations the real hand has (students / history policies via `scripts/export/
export_history_policy.py`); privileged information goes only into the critic, which is not exported. `teacher*` and
`arch_rma_s1` are NOT deployable: their actor reads the cube pose and noise-free joints.

## All runs

Status as of 2026-10-29 ~10:00. Eval = nominal / perturbed at the last checkpoint ([eval protocol](eval_protocol.md));
"best" = best checkpoint if better. Task ids are `orca-v2-inhand-repose<suffix>`.

| Run | Thread | Task id | Change vs baseline | Job / GPU | Status | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|---|---|---|---|---|
| A `firstrun_l40s` | [asym critic](asymmetric_critic.md) | `-v0` | baseline, 10k iters | 29823035 / L40S | done | 0.61 / 1.16 (best 1.37 / 1.62 @4500) | 11.6 / 14.6 (19.1 / 19.0) | 22 / 20 (33 / 25) | 1.8 / 2.2 (1.3 / 2.0) |
| B `abl_B_asym` | [asym critic](asymmetric_critic.md) | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | done | 5.66 / 6.55 | 66.3 / 62.0 | 86 / 52 | 0.20 / 0.98 |
| C `abl_C_asym_dr` | [asym critic](asymmetric_critic.md) | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | done | 6.08 / 6.40 (best 6.19 / 6.77 @4000) | 72.2 / 63.2 | 84 / 69 | 0.20 / 0.44 |
| `abl_B_asym_seed1` | [asym critic](asymmetric_critic.md) | `-asym-v0` | B, seed 1 | 29832230 / L40S | done | 5.23 / 5.92 (best 5.48 / 6.16 @2000) | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
| `abl_C_asym_dr_seed1` | [asym critic](asymmetric_critic.md) | `-asym-dr-v0` | C, seed 1 | 29838635 / A10 | done | 4.60 / 5.68 | 58.7 / 59.4 | 88 / 59 | 0.16 / 0.61 |
| sanity `sanity_nodr` | [sim2real](sim2real.md#sanity-runs-no-dr) | `-nodr-v0` | no DR, no obs noise (own no-DR eval) | 29829724 / A10 | done | 10.55 / 10.59 | 85.8 / 85.7 | 83 / 69 | 0.19 / 0.47 |
| `eval_sanity_nodr_on_dr` | [sim2real](sim2real.md#sanity-runs-no-dr) | eval on `-v0` | sanity `model_4999` evaluated with DR | 29836428 | done | 3.32 / 4.97 | 35.6 / 43.2 | 55 / 42 | 0.86 / 1.48 |
| `eval_abl_B_asym_ent5e-3_on_nodr` | [sim2real](sim2real.md#sanity-runs-no-dr) | eval on `-asym-nodr-v0` | ent5e-3 `model_4999` evaluated without DR | 29842299 | done | 8.58 / 8.07 | 87.8 / 79.7 | 95 / 84 | 0.05 / 0.23 |
| `ent5e-3_on_nodr` | [sim2real](sim2real.md#sanity-runs-no-dr) | eval on `-asym-nodr-v0` | ent5e-3 `model_4999` evaluated without DR | 29842299 / A10 | queued | | | | |
| `arch_asym_hist3` | [history](history_and_memory.md) | `-asym-hist3-v0` | B + MLP on 3-step history | 29831099 / A10 | done | 5.33 / 5.82 | 64.0 / 58.1 | 83 / 66 | 0.20 / 0.59 |
| `arch_asym_hist10` | [history](history_and_memory.md) | `-asym-hist10-v0` | B + MLP on 10-step history | 29831100 / A40 | done | 5.01 / 5.72 | 68.8 / 67.6 | 88 / 70 | 0.14 / 0.44 |
| `arch_asym_lstm` | [history](history_and_memory.md) | `-asym-lstm-v0` | B + LSTM actor-critic | 29831098 / A100 | done | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
| `teacher` | [teacher-student](teacher_student.md) | `-teacher-v0` | privileged teacher (not deployable), 10k | 29830926 / L40S | done | 7.44 / 7.63 | 81.4 / 74.4 | 94 / 73 | 0.06 / 0.31 |
| `teacher_cont18k` | [teacher-student](teacher_student.md) | `-teacher-v0` | teacher resumed from model_9999, +8k | 29840146 / 29840147 | cancelled (2026-10-28 16:25, superseded by `teacher_ent5e-3`) | | | | |
| `teacher_ent5e-3` | [teacher-student](teacher_student.md) | `-teacher-v0` | privileged teacher, entropy_coef 5e-3, 10k | 29842059 / L40S | done | 7.96 / 7.31 (best 8.16 @8000) | 79.5 / 68.1 | 88 / 72 | 0.13 / 0.44 |
| `teacher_rotlin_ent5e-3` | [teacher-student](teacher_student.md) | `-teacher-rotlin-v0` | privileged teacher on the rotlin reward, entropy 5e-3, 6k (not deployable) | 29846543 / L40S | done | **11.85** / 10.52 | 83.2 / 67.2 | **95** / 69 | 0.05 / 0.56 |
| `teacher_rotlin_cont10k_r6k` | [teacher-student](teacher_student.md) | `-teacher-rotlin-v0` + `--resume` from model_5999, ent 5e-3 | rotlin teacher continued 6k → 10k (still improving at 6k) | pod (W&B zc61lpl0) | done: no gain over 6k | 11.25 / 9.63 | 81.1 / 67.0 | 94 / 58 | 0.06 / 0.59 |
| `student_pt_rotlin_t10k_r6k` | [teacher-student](teacher_student.md) | `-student-pt-rotlin-v0`, 4096 envs, 2k | transformer student re-distilled from `teacher_rotlin_cont10k_r6k` (only if that teacher beats 11.85 RPM / 95 % at 6k) | pod | **dropped** (2026-10-30, +z-only decision: `student_pt_rotlin` stays the bidirectional fallback) | | | | |
| `student_pt_rotlin` | [teacher-student](teacher_student.md) | `-student-pt-rotlin-v0` | transformer student distilled from `teacher_rotlin_ent5e-3` model_5999 (4096 envs, 2k) | 29848449 / L40S (A100 and pod copies cancelled) | done | **10.90** / 9.81 | 78.9 / 67.0 | 86 / 56 | 0.17 / 0.64 |
| `student_lstm_rotlin` | [teacher-student](teacher_student.md) | `-student-lstm-rotlin-v0` | LSTM student from `teacher_rotlin_ent5e-3` | pod (`student_lstm_rotlin_r6k`, W&B 51frzpo6; DTU copy 29850539 cancelled) | done | **10.76** / 9.68 | 77.6 / 64.9 | **91** / 52 | 0.14 / 0.86 |
| `student_mlp_h10_rotlin` | [teacher-student](teacher_student.md) | `-student-mlp-h10-rotlin-v0` | MLP on 10-step history from `teacher_rotlin_ent5e-3` | pod (`student_mlp_h10_rotlin_r6k`, W&B 1uk7mra5; DTU copy 29850540 cancelled) | done | **11.09** / 9.96 | 78.6 / 65.0 | 81 / 50 | 0.27 / 0.69 |
| `teacher_onedir_droppen_ent5e-3` | [teacher-student](teacher_student.md) | `-teacher-onedir-droppen-v0`, ent 5e-3, 6k | privileged teacher on the demo reward (onedir +z, 3 rad/s clip, drop penalty), settings as `teacher_rotlin_ent5e-3` (not deployable) | 29851936 / L40S (W&B 742zn06f, artifact `teacher_onedir_droppen_ent5e-3`) | done | 15.19 / 14.41 (last-3 mean 15.51) | 79.4 / 69.1 | 69 / 17 (last-3 mean **74** / 17) | 0.45 / 2.89 |
| `student_pt_onedir_droppen` | [teacher-student](teacher_student.md) | `-student-pt-onedir-droppen-v0`, 4096 envs, 2k | transformer student from `teacher_onedir_droppen_ent5e-3` model_5999 | pod | **dropped** (user, 2026-10-30 13:30: one student only, the model_4500 one) | | | | |
| `student_pt_onedir_droppen_t4500` | [teacher-student](teacher_student.md) | same, `--checkpoint model_4500.pt` | student from the teacher's best single checkpoint (16.0 RPM / 81 % vs 15.2 / 69 % at 5999, a gap inside the eval noise): is the student sensitive to the teacher checkpoint | 29853994 / L40S (top of our L40S queue) | done: worse than B | 14.57 / 12.36 (last-3 mean 14.12) | 74.9 / 61.6 | 47 / 9 (last-3 mean 46 / 13) | 1.23 / 4.08 |
| `teacher_onedir_droppen_wristpen_ent5e-3` | [rewards](rewards.md#wrist-motion-2026-10-30) | `-teacher-onedir-droppen-wristpen-v0`, ent 5e-3, 6k | onedir teacher + wrist velocity penalty −4·q̇²(wrist): 742zn06f swings the wrist (range 1.2 rad, 2.2 rad/s mean vs 0.15 for the rotlin teacher) | pod (`..._r6k`, W&B cbgi1o9d) | done (teacher, not deployable) | 14.18 / 12.50 (last-3 mean 14.30) | 82.1 / 66.6 | **89** / 59 (last-3 mean **85** / **58**) | 0.11 / 0.59 |
| `teacher_onedir_droppen_acc10_ent5e-3` | [teacher-student](teacher_student.md) | `-teacher-onedir-droppen-v0` + `env.rewards.angular_acceleration.weight=10.0`, ent 5e-3, 6k | privileged teacher with the first +z win (cube smoothness ×10): smoother gait, hopefully easier to distill than 742zn06f | 29856919 / A100 | **cancelled** (entropy decision; default-entropy teacher instead) | | | | |
| `abl_B_onedir_droppen_wristpen_ent5e-3` | [rewards](rewards.md#wrist-motion-2026-10-30) | `-asym-onedir-droppen-wristpen-v0`, ent 5e-3 | same penalty on the deployable PPO config (vs the 3 onedir seeds) | 29856644 / A100 (-W 12h; + pod copy; the copy that starts second is cancelled) | **stopped** on the pod at start, DTU copy cancelled (entropy decision) | | | | |
| `abl_B_onedir_droppen_wristfix_ent5e-3` | [rewards](rewards.md#wrist-motion-2026-10-30) | `-asym-onedir-droppen-wristfix-v0`, ent 5e-3 | wrist not actuated (16-dim action, PD holds the default): the hard version of the wrist penalty, PT paper's hardware setting | 29856645 / A40 (-W 10.5h; + pod copy) | **cancelled** (entropy decision) | | | | |
| `student_mlp_h10` | [teacher-student](teacher_student.md) | `-student-mlp-h10-v0` | distill → MLP on 10-step history (old-reward teacher) | 29839014 / A100 | cancelled before starting (replaced by `student_mlp_h10_rotlin`) | | | | |
| `student_lstm` | [teacher-student](teacher_student.md) | `-student-lstm-v0` | distill → LSTM (old-reward teacher) | 29839015 / A40 | **crashed at load**: rsl_rl 3.1.2 builds a non-recurrent teacher MLP with the RNN's 256 inputs; fixed in `student_recurrent.py`, rerun on the rotlin teacher | | | | |
| `student_pt` | [teacher-student](teacher_student.md) | `-student-pt-v0` | distill → transformer + recon loss (4096 envs) | 29845112 / L40S | done | 6.89 / 7.52 | 75.2 / 71.3 | **95** / 66 | **0.05** / 0.56 |
| `arch_rma_s1` | [RMA](rma.md) | `-rma-teacher-v0` | B + RMA extrinsics encoder (not deployable) | 29835277 / L40S | done | 6.48 / 6.63 (best 6.55 / 7.03 @4000) | 74.5 / 69.3 | 89 / 66 | 0.16 / 0.61 |
| `arch_rma_s2` | [RMA](rma.md) | `-rma-student-v0` | RMA adaptation module, 30-step history, 2k | 29837730 / L40S | done | 6.20 / 6.71 (best 6.29 / 6.35 @1500) | 72.3 / 68.0 | 83 / 78 | 0.25 / 0.38 |
| `abl_B_rotlin` | [rewards](rewards.md) | `-asym-rotlin-v0` | B, linear clipped rotation reward + off-axis | 29838450 / L40S | done | **10.41** / 11.07 | 74.0 / 67.8 | 84 / 44 | 0.22 / 1.09 |
| `abl_B_rotlin_cont10k` | [rewards](rewards.md) | `-asym-rotlin-v0` + `--resume` from `abl_B_rotlin` model_4999 | rotlin trained to 10k (still improving at 5k) | 29850584 / A10 (A100 copy 29846685 cancelled) | done | 10.81 / 10.70 | 77.7 / 67.4 | **91** / 55 | 0.12 / 0.77 |
| `abl_B_rotlin_seed1_r6k` | [rewards](rewards.md) | `-asym-rotlin-v0 --seed 1` | rotlin, second seed (pod; resumed under MPS from model_2000, W&B 1jaodg4h) | pod | done | **10.25** / 11.48 | 76.8 / 71.4 | 83 / 45 | 0.23 / 1.08 |
| `abl_B_droppen_seed1_r6k` | [rewards](rewards.md) | `-asym-droppen-v0 --seed 1` | drop penalty, second seed (pod; resumed under MPS from model_2000, run dir `2026-10-29_20-56-39_abl_B_droppen_seed1_resume_r6k`, W&B 3o9mpo0o) | pod | done | 7.00 / 7.55 (best 7.06 @3000) | 80.3 / 74.8 | 88 / 58 (best 92 @3500) | 0.14 / 0.75 |
| `abl_B_onedir_droppen_clip2_ent5e-3_r6k` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.rotation_progress.params.max_vel=2.0` | onedir + drop penalty, speed reward clipped at 2 rad/s instead of 3 | pod | done | 10.09 / 9.68 | 76.3 / 59.9 | 91 / 41 | 0.11 / 1.36 |
| `abl_B_onedir_droppen_clip2_ent5e-3_seed1` | [rewards](rewards.md) | `-asym-onedir-droppen-v0 --seed 1` + max_vel 2.0 | second seed of the 2 rad/s cap | 29851783 / A40, 29851823 / A100 | **cancelled** (never started): 2 rad/s is dominated by rotlin 10k | | | | |
| `abl_B_rotlin_hist3` | [history](history_and_memory.md) | `-asym-rotlin-hist3-v0` | rotlin + 3-step history MLP (entropy default, vs abl_B_rotlin 10.33) | 29851782 / L40S | done | 10.49 / 10.48 | 71.2 / 59.6 | 81 / 34 | 0.27 / 1.30 |
| `abl_B_rotlin_work` | [rewards](rewards.md#work-penalty) | `-asym-rotlin-work-v0` (weight −1.5) | rotlin + HORA work penalty −1.5·Σ\|τ·q̇\| | 29851888 / L40S | **killed** at it 467: learned to drop the cube to stop the penalty (100 % drops, rotation 1.7 vs 25) | | | | |
| `abl_B_rotlin_work015` | [rewards](rewards.md#work-penalty) | `-asym-rotlin-work-v0` | rotlin + HORA work penalty −0.15·Σ\|τ·q̇\| (entropy default, vs abl_B_rotlin 10.33) | 29851938 / L40S | done | 10.02 / 10.32 | 69.2 / 61.8 | 86 / 47 | 0.14 / 0.89 |
| `abl_B_rotlin_ent5e-3_env16k_r6k` | [PPO / capacity](ppo_and_capacity.md) | `-asym-rotlin-v0 --num_envs 16384` | rotlin + ent 5e-3 with 2× envs (2× samples per iteration, same PPO settings) | pod | done | 9.47 / 8.76 | 73.4 / 59.1 | 67 / 27 | 0.38 / 1.64 |
| `abl_B_rotlin_ent5e-3_offaxis8_r6k` | [rewards](rewards.md) | `-asym-rotlin-v0` + `env.rewards.off_axis_angular_velocity.weight=-8.0` | rotlin + ent 5e-3, off-axis −8 (−5 works, −15 stops rotating) | pod, after the seeds (~01:00Z) | **stopped** at it ~2252 (checkpoints to model_2000 kept, no eval) on the pod (2026-10-30, +z-only decision) | | | | |
| `abl_B_rotlin_ent5e-3` | [rewards](rewards.md) | `-asym-rotlin-v0` | rotlin + entropy 5e-3 | 29845927 / A100 | done | 9.50 / 10.43 | 76.1 / 70.8 | 81 / 58 | 0.27 / 1.00 |
| `abl_B_rotlin_droppen_ent5e-3` | [rewards](rewards.md) | `-asym-rotlin-droppen-v0` | rotlin + ent 5e-3 + drop penalty −100 return | 29845937 / A40 | done | 8.29 / 8.17 (best 8.37 @2000) | 70.8 / 60.8 | **89** / 44 | 0.17 / 0.92 |
| `abl_B_rotlin_ent5e-3_offaxis15` | [rewards](rewards.md) | `-asym-rotlin-v0` + `env.rewards.off_axis_angular_velocity.weight=-15.0` | rotlin + ent 5e-3, off-axis penalty −5 → −15 | 29846126 / A10 | **done: stops rotating** | 0.19 / 0.26 | 37.0 / 31.8 | 91 / 80 | 0.13 / 0.22 |
| `abl_B_rotlin_ent1e-2` | [PPO / capacity](ppo_and_capacity.md) | `-asym-rotlin-v0` | rotlin + entropy 1e-2 | 29846596 / A10 | **killed at it 640: does not learn** (action std 1.8 → 3.0, episodes ~200 steps, 95 % drops; B + ent 1e-2 had 450 steps / 40 % at it 600) | | | | |
| `abl_B_rotlin_ent5e-3_big` | [PPO / capacity](ppo_and_capacity.md) | `-asym-rotlin-v0` + actor / critic `[1024,1024,512,256]` | rotlin + ent 5e-3, 2× wider MLPs (~4× parameters; default [512,512,256,128]) | 29846127 / A40 (eval 29852090 / A10) | done | 10.08 / 10.97 | 72.1 / 67.2 | 78 / 62 | 0.27 / 0.59 |
| `abl_B_onedir_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-v0` (unclipped) | B + ent 5e-3, +z only at 1.5 rad/s, unclipped linear speed reward | 29845346 / L40S | **killed at it 272: reward hack** (flings the cube, 99.9 % drops) | | | | |
| `abl_B_onedir_clip3_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-v0` | B + ent 5e-3, +z only, linear speed reward clipped at 3 rad/s (eval +z only) | 29846083 / L40S | done | **15.19** / 13.97 (best 15.67 @1000) | 79.2 / 65.9 | 44 / 9 | 1.30 / 4.83 |
| `abl_B_onedir_droppen_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` | onedir (+z, clip 3 rad/s) + drop penalty + ent 5e-3 | 29849218 / A100 | done | **16.41** / 15.07 | 82.8 / 69.9 | 63 / 19 (best 73 @1000) | 0.64 / 2.94 |
| `abl_B_onedir_droppen6k_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.object_drop.weight=-6000` | onedir (+z, clip 3 rad/s) + 3× drop penalty (−300 return) + ent 5e-3: the other lever against drops besides the 2 rad/s cap, keeps the speed headroom | pod (`abl_B_onedir_droppen6k_ent5e-3_r6k`, W&B 7ogu7mc5) | done: no gain | 14.38 / 14.12 (last-3 mean 14.48) | 77.6 / 68.2 | 53 / 17 (last-3 mean 54) | 1.14 / 3.94 |
| `abl_B_onedir_droppen_ent5e-3_seed1` | [rewards](rewards.md) | `-asym-onedir-droppen-v0 --seed 1`, ent 5e-3 | second seed of the fastest config (onedir 3 rad/s + drop penalty, 16.41 RPM) | 29852087 / A40 (pod copy skipped) | done | 14.89 / 14.40 (last-3 mean 14.81) | 78.7 / 67.1 | 41 / 14 (last-3 mean 46 / 15) | 1.11 / 4.36 |
| `abl_B_onedir_droppen_ent5e-3_seed2` | [rewards](rewards.md) | `-asym-onedir-droppen-v0 --seed 2`, ent 5e-3 | third seed of the onedir baseline, so the single-knob variants (6k penalty, acc10, pos45, cap 4) can be judged against its seed spread | 29852807 / A10 | **stopped** at it ~2160 (entropy decision 2026-10-30 18:30) | | | | |
| `abl_B_onedir_droppen_ent5e-3_cont10k` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `--resume` from `abl_B_onedir_droppen_ent5e-3` model_4999, ent 5e-3 | fastest config trained to 10k (still +0.9 RPM from 4k to 5k; rotlin gained +0.4 RPM / +6 DFSR from 5k to 10k) | 29852088 / A100 | done | **16.88** / 14.22 (last-3 mean **16.89**) | 83.6 / 66.4 | 59 / 14 (last-3 mean 55 / 12) | 0.70 / 3.41 |
| `abl_B_onedir_droppen_clip4_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.rotation_progress.params.max_vel=4.0` | speed cap 4 rad/s: next point on the speed / drop curve (cap 2 → 10.1 RPM, cap 3 → 16.4 RPM) | 29852091 / A40 (pod copy skipped) | done: no gain | 16.18 / 15.47 (last-3 mean 16.03) | 79.1 / 68.7 | 45 / 13 (last-3 mean 50 / 12) | 1.12 / 3.83 |
| `abl_B_onedir_droppen_acc10_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.angular_acceleration.weight=10.0` | cube smoothness term (exp of angular acceleration, σ 0.5) ×10: against spin-up drops (~1/3 of drops in the first 3 s) | pod (`abl_B_onedir_droppen_acc10_ent5e-3_r6k`, W&B mau1bbmt) | done: **better** | 14.97 / 14.36 (last-3 mean 14.96) | 82.3 / 70.6 | **72** / 33 (last-3 mean **70** / 26) | 0.53 / 1.86 |
| `abl_B_onedir_droppen_acc10_ent5e-3_seed1` | [rewards](rewards.md) | as acc10 + `--seed 1` | replicate of the first +z win | 29856909 / A10 (+ pod copy; the copy that starts second is cancelled) | **cancelled** (entropy decision; acc10 retested on default entropy) | | | | |
| `abl_B_onedir_droppen_acc30_ent5e-3_r6k` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.angular_acceleration.weight=30.0` | cube smoothness ×30: dose response | pod | **dropped** (entropy decision) | | | | |
| `abl_B_onedir_droppen_ramp_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-ramp-v0` | speed-reward cap ramps 0.5 → 3 rad/s over the first 3 s of each episode: against spin-up drops via the reward (acc10 does it via smoothness) | pod (`..._r6k`, W&B ty9c9smm) | done | 15.92 / 14.98 (last-3 mean 16.05) | 84.8 / 73.4 | 66 / 27 (last-3 mean 69 / 24) | 0.44 / 2.16 |
| `abl_B_onedir_droppen_ent1e-4_r6k` | [PPO / capacity](ppo_and_capacity.md#action-noise-under-the-entropy-bonus) | `-asym-onedir-droppen-v0` (default entropy 1e-4) | control: the onedir line had only run with entropy 5e-3 | pod (W&B qn0pkisu) | done: **best +z** | **17.66** / 15.39 (last-3 mean **17.53**) | 89.9 / 78.3 | **80** / 36 (last-3 mean **76** / 28) | 0.30 / 1.91 |
| `abl_B_onedir_droppen_ent1e-4_seed1` | [rewards](rewards.md) | `-asym-onedir-droppen-v0 --seed 1` (default entropy) | replicate of the new +z baseline | pod (W&B cule0dej, since 17:24Z; DTU copy 29856937 cancelled) | running | | | | |
| `abl_B_onedir_droppen_ent1e-4_seed2` | [rewards](rewards.md) | `--seed 2` (default entropy) | third seed of the new baseline | 29856942 / A10 + pod copy | **cancelled** before starting (2026-10-30 19:25: tests a tumbling config; replaced by upright variants) | | | | |
| `abl_B_onedir_droppen_ent1e-4_10k` | [rewards](rewards.md) | `-asym-onedir-droppen-v0`, default entropy, 10k from scratch | longer training on the new baseline (entropy 5e-3: 10k was +1.2 RPM, not safer); its 5k checkpoint doubles as a same-seed replicate | 29857017 / L40S | **cancelled** before starting (2026-10-30 19:25: tests a tumbling config; replaced by upright variants) | | | | |
| `abl_B_onedir_droppen_upright_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | `-asym-onedir-droppen-upright-v0` (default entropy) | cube tilt penalty −200·(1 − max_i \|R e_i · z\|): flat z-rotation instead of tumbling | pod (W&B 5oklh054, since 18:43Z; DTU copy 29857090 cancelled) | running | | | | |
| `abl_B_onedir_droppen_upright_wristfix_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | `-asym-onedir-droppen-upright-wristfix-v0` (default entropy) | tilt penalty + wrist not actuated (2×2 with baseline / wristfix / upright) | 29857091 / A40 + pod copy | queued | | | | |
| `abl_B_onedir_droppen_upright500_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | upright + `env.rewards.cube_tilt.weight=-500.0` | stronger tilt penalty (dose) | 29857222 / L40S + pod copy | queued | | | | |
| `abl_B_onedir_droppen_upright_wristpen_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | `-asym-onedir-droppen-upright-wristpen-v0` | tilt penalty + wrist velocity penalty (soft alternative to the fixed wrist) | 29857223 / A100 (pod copy skipped) | running (since 23:10) | | | | |
| `abl_B_onedir_droppen_upright_ent1e-4_seed1` | [rewards](rewards.md#cube-tumbling-2026-10-30) | upright `--seed 1` | replicate of the tilt penalty | 29857224 / A10 (pod copy skipped) | running (since 20:45) | | | | |
| `abl_B_onedir_droppen_upright_wristfix_ent1e-4_seed1` | [rewards](rewards.md#cube-tumbling-2026-10-30) | upright + fixed wrist `--seed 1` | replicate of the likely demo config (queued ahead; cancelled if the tilt penalty does not keep the cube flat) | 29857428 / A10 | queued | | | | |
| `teacher_onedir_droppen_upright_wristfix` | [teacher-student](teacher_student.md) | `-teacher-onedir-droppen-upright-wristfix-v0`, default entropy, 6k | privileged +z teacher with the tilt penalty and the wrist fixed (queued ahead, same condition); student task `-student-pt-onedir-droppen-upright-wristfix-v0` registered | 29857427 / A100 | queued | | | | |
| `abl_B_onedir_droppen_acc10_ent1e-4` | [rewards](rewards.md#smoothness-on-default-entropy-2026-10-31) | + `env.rewards.angular_acceleration.weight=10.0` (default entropy) | both +z wins together: does smoothness ×10 still add on default entropy | pod (W&B ps0bke4n; DTU copy 29856938 cancelled) | done: **robustness ↑ (perturbed 53 vs 28 %), still tumbling** (tilt 40°, 200 flips/min) | 16.46 / 15.28 (last-3 mean 15.69) | 87.6 / 76.8 | 88 / 55 (last-3 mean 82 / **53**) | 0.14 / 1.02 |
| `teacher_onedir_droppen_ent1e-4` | [teacher-student](teacher_student.md) | `-teacher-onedir-droppen-v0`, default entropy, 6k | the +z teacher without the 5e-3 bonus (742zn06f had std ~3): better teacher, in-range actions, easier to distill? | 29856939 / A100 + pod copy | **dropped** (2026-10-30 19:00: the +z policies tumble the cube; the next teacher comes from the winning upright / wrist variant) | | | | |
| `abl_B_onedir_droppen_wristpen_ent1e-4` | [rewards](rewards.md#wrist-motion-2026-10-30) | `-asym-onedir-droppen-wristpen-v0` (default entropy) | wrist velocity penalty on the new baseline | 29856940 / A100 + pod copy | **cancelled** before starting (2026-10-30 19:25: tests a tumbling config; replaced by upright variants) | | | | |
| `abl_B_onedir_droppen_wristfix_ent1e-4` | [rewards](rewards.md#wrist-motion-2026-10-30) | `-asym-onedir-droppen-wristfix-v0` (default entropy) | wrist not actuated, on the new baseline | 29856941 / A40 + pod copy | queued | | | | |
| `abl_B_onedir_droppen_alpha05_std05_ent1e-4_r6k` | [action space](action_space.md) | + `env.actions.joint_pos.alpha=0.5 agent.policy.init_noise_std=0.5` (default entropy) | EMA α 0.5 on the new baseline | 29857018 / A10 + pod copy | **cancelled** before starting (2026-10-30 19:25: tests a tumbling config; replaced by upright variants) | | | | |
| `abl_B_onedir_droppen_ent5e-3_anneal0` | [PPO / capacity](ppo_and_capacity.md#action-noise-under-the-entropy-bonus) | `-asym-onedir-droppen-v0` + `--resume` from `abl_B_onedir_droppen_ent5e-3` model_4999, entropy 0, +2k | does removing the bonus late (std 3 → ?) improve the deterministic policy | 29856764 / L40S (moved from A100, -W 6h) | **cancelled** before starting (entropy decision) | | | | |
| `abl_B_onedir_droppen_pos45_ent5e-3` | [rewards](rewards.md) | `-asym-onedir-droppen-v0` + `env.rewards.object_position_tracking.weight=45.0` | cube position term (σ 2 cm) ×3: keep the cube centred on the palm | 29852803 / L40S (A40 copy 29852318 and pod copy cancelled) | done: **no tumbling, most robust** | 12.59 / 13.77 (last-3 mean 12.79) | 80.9 / 74.3 | **86** / **50** (last-3 mean **88** / **49**) | 0.14 / 0.97 |
| `abl_B_onedir_droppen_pos45_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | position ×3, default entropy | the non-tumbling, robust config on the better entropy | pod, 4000 it (W&B kfuq4jwb, since 00:49Z; DTU copy 29857658 cancelled) | running | | | | |
| `abl_B_onedir_droppen_upright_acc10_ent1e-4` | [rewards](rewards.md#smoothness-on-default-entropy-2026-10-31) | `-asym-onedir-droppen-upright-v0` + `env.rewards.angular_acceleration.weight=10.0`, default entropy, 4000 it | tilt penalty (against the tumbling) + smoothness ×10 (perturbed robustness 28 → 53 %) | 29858486 / L40S (-W 8h) | queued | | | | |
| `abl_B_onedir_droppen_upright_pos45_ent1e-4` | [rewards](rewards.md#cube-tumbling-2026-10-30) | `-asym-onedir-droppen-upright-v0` + position ×3, default entropy | tilt penalty and position ×3 together | 29857659 / L40S + pod copy (4000 iterations on the pod for the 09:00Z deadline) | queued | | | | |
| `abl_B_onedir_droppen_pos45_adr_from5e-3` | [sim2real](sim2real.md#adr-on-action-delay-and-encoder-offset-2026-10-30) | `-asym-onedir-droppen-adr-v0` + position ×3, `--resume` from `abl_B_onedir_droppen_pos45_ent5e-3` model_4999, default entropy, +3k | the robust non-tumbling policy made deployment-robust: ADR on action delay (0–15 substeps at L = 1) and encoder offset (±0.08 rad at L = 1) | pod (since ~22:20Z; DTU copy 29857697 cancelled) | running | | | | |
| `abl_B_rotsoft` | [rewards](rewards.md) | `-asym-rotsoft-v0` | B, magnitude σ 0.75, mag 60 / dir 30 | 29838451 / A100 | done — **reward hack** (wobbles, doesn't rotate) | −0.02 / −0.01 | 6.8 / 5.8 | 92 / 77 | 0.09 / 0.34 |
| `abl_B_droppen` | [rewards](rewards.md) | `-asym-droppen-v0` | B + −100 return on drop | 29835278 / A40 | done | **7.94** / 8.36 (best 8.34 / 8.23 @3000) | 80.0 / 77.2 | 83 / 61 (best 94 @3000) | 0.27 / 0.55 |
| `abl_B_asym_ent5e-3` | [PPO / capacity](ppo_and_capacity.md) | `-asym-v0` | B + entropy_coef 5e-3 | 29833758 / A100 | done | **7.33 / 7.31** | 78.6 / 69.8 | 88 / 55 | 0.12 / 0.75 |
| `abl_B_asym_ent5e-3_seed1` | [PPO / capacity](ppo_and_capacity.md) | `-asym-v0` | ent5e-3, seed 1 | 29841912 / A100 | done | 5.74 / 5.45 | 69.9 / 59.0 | 88 / 56 | 0.13 / 0.55 |
| `abl_B_asym_ent1e-2` | [PPO / capacity](ppo_and_capacity.md) | `-asym-v0` | B + entropy_coef 1e-2 | 29841914 / A100 | done | 7.60 / 7.47 (best 7.69 / 7.65 @4000) | 79.4 / 69.7 | 75 / 48 | 0.30 / 0.88 |
| rotlin + drop penalty | [rewards](rewards.md) | – | rotlin + −200 drop weight (−10 return) | – | planned | | | | |
| position / velocity reward | [rewards](rewards.md) | – | position σ 0.05, weight 5, + cube lin-vel penalty | – | planned | | | | |
| `abl_B_rotlin_ent5e-3_alpha05` | [action space](action_space.md) | `-asym-rotlin-v0` + `env.actions.joint_pos.alpha=0.5` | rotlin + ent 5e-3, EMA α 0.2 → 0.5 | 29849219 / A10 | **killed at it 224: no learning** (4-step episodes, 100 % drops: the initial noise std 1.5 knocks the cube off at once) | | | | |
| `abl_B_rotlin_ent5e-3_alpha05_std05` | [action space](action_space.md) | as above + `agent.policy.init_noise_std=0.5` | α 0.5 with calmer initial exploration (std 1.5 → 0.5) | 29850659 / A10 | **stopped** at it ~850 (2026-10-30, +z-only decision; replaced by the onedir version) | | | | |
| `abl_B_onedir_droppen_ent5e-3_alpha05_std05` | [action space](action_space.md) | `-asym-onedir-droppen-v0` + `env.actions.joint_pos.alpha=0.5 agent.policy.init_noise_std=0.5`, ent 5e-3 | EMA α 0.5 (faster target tracking) with calmer initial exploration, on the +z demo config | pod (`..._r6k`, W&B ugbqegtv, since 17:09Z; DTU copy 29853007 cancelled) | **stopped** on the pod at it ~100 (entropy decision) | | | | |
| `abl_B_onedir_droppen_work015_ent5e-3` | [rewards](rewards.md#work-penalty) | `-asym-onedir-droppen-work-v0`, ent 5e-3 | +z demo config + work penalty −0.15 (on rotlin: same speed, perturbed drop-free 56 vs 39 % over the last 3 checkpoints) | 29853308 / L40S (pod copy skipped) | done: robustness ↑, still tumbling | 14.35 / 13.77 (last-3 mean 14.75) | 75.3 / 66.8 | 69 / 44 (last-3 mean 65 / **45**) | 0.42 / 1.20 |
| `abl_B_rotlin_ent5e-3_relact` | [action space](action_space.md) | `-asym-rotlin-relact-v0` | rotlin + ent 5e-3, relative joint targets (delta 0.1 half-range / step; the B-reward version 29845614 was cancelled before starting) | 29845929 / A10 | **killed at it 96**: hand cannot close from the reset pose (6-step episodes); needs grasp-cache resets | | | | |
| `student_mlp_h10_ft` | [student fine-tune](student_finetune.md) | `-student-mlp-h10-ft-v0` (fallback `-student-mlp-h10-res-v0`) | PPO fine-tuning of the MLP-h10 student: teacher critic, 150-iter critic warm-up, kickstarting β 1 → 0 | – | implemented + CPU-tested; launch after `student_mlp_h10` | | | | |
| fixed-delay run | [sim2real](sim2real.md#fixed-action-delay-sim--deploy) | – | fixed 1-step action delay + next-tick actuation (Phase 2) | – | planned | | | | |
| `acthold` | [sim2real](sim2real.md#random-action-holding) | `-asym-acthold-v0` | B + random action holding (TeleDexter's action masking): joint targets not updated w.p. p ~ U(0, 0.1) per step (lost / late servo packet); obs = command sent; with the fixed delay in Phase 2 | – | implemented + CPU-tested; Phase 2 | | | | |
| `actnet` | [actuator net](actuator_net.md) | `-asym-actnet-v0` | B + chirp-fitted servo model (delay, play, τ, learned target correction; arXiv 2603.04351 / Hwangbo 2019). Offline on v1b chirps: position RMSE 2.20° → 1.38°, phase error 18.6° → 8.0° vs the current PD | – | implemented + CPU-tested; retrain on v2 chirps, Phase 2 | | | | |

Cancelled before starting (2026-10-27): A-based `-hist3` / `-hist10` / `-lstm` (29830927 / 28 / 30). Registered but
parked sim2real tasks (`-delay`, `-jointbias`, `-handgrav`, `-wristfix`, `-droppen`, `-hist3`, `-asym-acthold`): [sim2real](sim2real.md#registered--planned-sim2real-runs-parked).
Training rewards, episode lengths and drop rates per run are in the topic files.

## Topic files

| File | Contents |
|---|---|
| [eval_protocol.md](eval_protocol.md) | metrics (RPM, RA, DFSR, DC), nominal / perturbed eval, checkpoint selection, eval pipeline |
| [asymmetric_critic.md](asymmetric_critic.md) | A baseline, B, C, seeds |
| [history_and_memory.md](history_and_memory.md) | hist3, hist10, asym LSTM |
| [teacher_student.md](teacher_student.md) | privileged teacher, continuation, MLP-h10 / LSTM / PT students, distillation |
| [student_finetune.md](student_finetune.md) | PPO fine-tuning of the distilled student |
| [rma.md](rma.md) | RMA stage 1 / 2, HORA port |
| [rewards.md](rewards.md) | reward literature, diagnosis, rotlin, rotsoft, drop penalty, planned variants |
| [ppo_and_capacity.md](ppo_and_capacity.md) | entropy run, bigger network, PPO hyperparameter ideas |
| [action_space.md](action_space.md) | EMA α, relative actions, control frequency |
| [export.md](export.md) | TorchScript export of history policies (student_pt, RMA s2, hist MLPs, LSTM/GRU) with the history inside the module; deploy contract |
| [actuator_net.md](actuator_net.md) | Phase 2: learned servo model from chirp recordings (delay, play, τ, residual net); offline results, retraining on v2 chirps |
| [sim2real.md](sim2real.md) | Phase 2 (parked): delay, joint bias, gravity, wrist, DR ranges, actuator models, sanity runs, deploy facts |
| [literature.md](literature.md) | PT paper takeaways, general literature reviews, paper → run mapping, code availability |
| [infrastructure.md](infrastructure.md) | DTU LSF queues, limits, quota / pruning, hanging jobs, submit scripts |
| [changelog.md](changelog.md) | dated one-line timeline |

## Running and evaluating

Variants are task ids in `tasks/orca_inhand/orca_v2_repose_ablation_env_cfg.py` (seed 42, 8192 envs, baseline PPO cfg
unless noted). Submit with `hpc/submit_ablation.sh TASK RUN_NAME ITERS QUEUE WALLTIME [EXTRA]`; the job trains, then runs
the nominal + perturbed eval on 5 checkpoints, renders MuJoCo videos and uploads to W&B (`rwr-fast/orca_v2_repose`).
The job scripts are not in git. Details: [infrastructure](infrastructure.md), [eval protocol](eval_protocol.md).
