# Hyperparameter exploration notes — orca-v2-inhand-repose-v0

Baseline: `firstrun_l40s` (W&B `rwr-fast/orca_v2_repose`, run `ffueb4yh`), 10k iters, 8192 envs, L40S ~4.3 s/iter (~12 h).
Reward plateaued at ~490–510 from iter ~4000; ~60 % of episodes end in a cube drop; mean episode length ~395 / 640 steps.

## First: what we do NOT model in training

- **No actuator noise / latency model.** Actions go straight to an ideal implicit PD drive every physics step:
  no command delay, no action noise, no backlash, no servo bandwidth limit beyond the PD gains themselves.
  Only the PD gains are randomized (stiffness ×0.75–1.5, damping ×0.3–3.0 per reset).
- **No realistic observation noise.** The only corruption is white Gaussian noise on `joint_pos` (std 0.10 on the
  limit-normalized value) and `joint_vel` (std 0.05). No observation delay, no per-episode encoder bias/offset,
  no quantization, no dropped packets.
- **No soft fingertips.** Fingers are rigid bodies with hard PhysX contacts; high friction (0.8–2.0) is the only
  stand-in for grippy pads (no contact patch / extra torsional grip).

The real hand will have all of these, so these are the first sim2real gaps to close.

## 1. Sim2real: actuator + observation models

| Idea | Where | Values to try |
|---|---|---|
| Action delay (hold previous command) | action term / custom wrapper | 0, 1, 2 control steps (0 / 50 / 100 ms), sampled per episode |
| Action noise | `action_noise_model` in env cfg | Gaussian std 0.01–0.05 on joint targets |
| Observation delay | obs term with history buffer | 0–1 control steps, sampled per episode |
| Per-episode joint offset (calibration error) | `observation_noise_model` / custom | bias ±0.02–0.05 rad |
| Measure real latency + noise first | Subtask 1 chirps (Section 7) | fit the above ranges to the measurements |

## 2. Domain randomization ranges

| Parameter | Current | To explore |
|---|---|---|
| Joint stiffness / damping scale | 0.75–1.5 / 0.3–3.0 | re-center on Subtask 1 sysid values |
| Cube mass scale | 0.5–1.5 (110 g) | keep; match real cube mass |
| Cube friction | 0.3–1.0 | match real cube material |
| Cube size | fixed 45 mm | ±5 % if the real cube differs |
| External force/torque pushes on cube | none | small random impulses → recovery behaviour |
| Initial joint positions | effectively none (`reset_robot_joints` scales 0 by 0.2) | ±0.1–0.2 rad around default |
| Scene gravity noise | gaussian **mean −0.1**, std 0.1 (params are (mean, std), likely a typo) | mean 0, std 0.1 or uniform ±0.1 |
| Fingertip softness | **not modeled**: rigid links, (likely) convex-hull colliders, PhysX hard contacts; only high friction 0.8–2.0, restitution 0 as a proxy | PhysX compliant contacts on the finger/fingertip material: `RigidBodyMaterialCfg(compliant_contact_stiffness=…, compliant_contact_damping=…)`, randomize stiffness per env; tune to real pad compliance (no full soft-body sim, too slow at 8192 envs) |

## 3. Architecture / observations

| Idea | Change | Notes |
|---|---|---|
| **Asymmetric critic** (try first) | `critic: TeacherObsGroupCfg = TeacherObsGroupCfg()` in `ObservationsCfg` | critic sees noise-free cube pose; deployed actor unchanged |
| Add cube lin/ang vel to critic | uncomment in `TeacherObsGroupCfg` | |
| Proprioceptive history | stack last 3–10 obs, or GRU actor (`ActorCriticRecurrent`) | lets the policy infer cube state from contacts |
| Teacher–student distillation | `DistillationRunner` | only if the two above stall |
| Network size | [512, 512, 256, 128] | [256, 256, 128] (faster) vs [1024, 512, 256] |

## 4. Rewards

Current: direction 60, magnitude 30 (σ 0.2), position 15 (σ 0.02), ang. acc 1, joint_vel −1e-5, action −2e-4, action_rate −0.075.

| Idea | Values |
|---|---|
| Explicit drop penalty / termination reward | −50 … −200 on `object_dropped` |
| Position tracking σ | 0.02 → 0.03 (looser) |
| Action-rate penalty (smoothness for the real servos) | −0.075 → −0.15 |
| Magnitude vs direction weighting | 30/60 → 45/45 |

## 5. PPO hyperparameters

Current: lr 1e-4 (adaptive, desired_kl 0.01), gamma 0.99, lambda 0.95, 24 steps/env, 5 epochs × 4 minibatches,
entropy 1e-4, clip 0.2, init_noise_std 1.5.

| Parameter | Values to try | Why |
|---|---|---|
| `entropy_coef` | 1e-3, 5e-3 | noise std collapsed to 0.18 → little exploration after ~4k iters |
| `gamma` | 0.99 → 0.995 | drops are delayed consequences; 0.99 horizon ≈ 5 s at 20 Hz |
| `num_steps_per_env` | 24 → 32/48 | longer rollouts, better advantage estimates |
| `desired_kl` | 0.01 → 0.016 | faster learning early |
| `num_envs` | 8192 vs 16384 | more data per iteration (fits on L40S?) |
| `max_iterations` | 4000–6000 is enough for screening runs | baseline plateaued at ~4k |

## 6. Control / env

| Parameter | Current | To explore |
|---|---|---|
| EMA action smoothing `alpha` | 0.2 | 0.3, 0.5 (faster response) |
| Control frequency | 20 Hz (decimation 6 @ 120 Hz) | keep unless the real loop runs faster |
| Spin command range | 0.9–1.5 rad/s, resampled every 8 s | curriculum: start slow, increase |
| Episode length | 32 s | keep |

## 6b. Fixed action delay (sim + deploy)

Deploy code can only **add** delay, never remove the inherent one (inference, ROS hop, servo bus, servo's internal loop).
So make the total delay **fixed and known** and train for exactly that:

- **Deploy (next-tick actuation):** in the policy node (`rwr_faive_system/experiments/rl_inference/policy_inference_cube_spin_v2.py`,
  not released yet), read obs at tick t, run the policy, and send the command **exactly at tick t+1** instead of as soon as
  inference finishes. Total delay = 1 step (50 ms) + servo lag; inference/ROS jitter is absorbed because 50 ms > their time.
  Optional `action_delay_steps` parameter that buffers k commands.
- **Sim:** action term keeps a FIFO of the last k processed actions and applies the one from k steps ago (k = 1 fixed,
  or sampled per episode from {1, 2} to cover the servo-side variation).
- **Observation consistency:** `last_action` / `last_joint_pos_cmd` in the 74-dim obs must be the action the policy
  **produced** (not the delayed one applied), identically in sim and on deploy. With k = 1 the last action is exactly
  the pending command, so the state stays Markov.
- EMA smoothing (alpha 0.2) must be applied at the same point in the chain in sim and deploy (before the delay buffer).

Steps:
1. Estimate the inherent delay from existing chirp CSVs (`scripts/motor_param_characterization/202506_collected_realworld_data`,
   `202510_orcav1b_realworld_data`; 40 Hz, older hand → rough) via phase lag vs frequency; redo with v2 chirps at 20 Hz.
2. Add `action_delay_steps` (fixed / per-episode random) to the action term as a new ablation task id.
3. When `rwr_faive_system` is released: next-tick actuation in the policy node (a few lines).

## 7. Takeaways from the ORCA Proprioceptive Transformer paper (arXiv 2605.21330, ETH SRL, IROS 2026)

Their codebase is the base of ours: same Isaac Lab setup (8192 envs, 120 Hz / 20 Hz, 32 s episodes), same teacher
MLP [512, 512, 256, 128], same reward weights (direction 60, magnitude 30, action rate −0.075), same 81-dim teacher obs
(= our `TeacherObsGroupCfg`), teacher 10k PPO iters at lr 1e-4.

**Our hardware: plain ORCA hand, no joint sensors → proprio from motor encoders only** = their *PT-Motor* setting.

Real-hand results (55 mm cube, 3 × 60 s trials):

| Policy | RPM | Rot. accuracy | Drop-free success | Drops |
|---|---|---|---|---|
| Proprio-PPO (= our run A: MLP, single noisy obs) | 3.83 ± 0.51 | 92.6 ± 10.5 % | 89.7 ± 14.5 % | 0.33 |
| PT-Motor (transformer student, motor encoders) | 9.33 ± 0.63 | 96.2 ± 6.6 % | 98.1 ± 3.3 % | 0.33 |
| PT-Joint (joint sensors; **not available to us**) | 11.83 ± 0.52 | 100 % | 100 % | 0 |

Student ablation with motor encoders only and a 10-step history (our case):

| Student | RPM | Drop-free success | Drops |
|---|---|---|---|
| MLP | 5.08 | 61.5 % | 2.0 |
| LSTM | 7.00 | 100 % | 0 |
| Transformer | 9.33 | 98.1 % | 0.33 |
| Transformer **without reconstruction loss** | 4.75 | 35.0 % | 5.0 |

Takeaways:

1. **Run A is the weak baseline.** Expect ~4 RPM (~0.4 rad/s vs 1.5 rad/s commanded) and high variance on hardware.
2. **The deployable gain is teacher → history student, not the critic.** B/C only improve training of a policy that
   still sees one noisy obs. Plan: privileged teacher (teacher obs as *actor* input) → distill to a student on joint
   history. The repo's `DistillationRunner` / teacher–student leftovers are the intended path.
3. **History length matters most for motor-only sensing.** Window 3 with motor encoders dropped to 73.9 % rot. accuracy;
   use **window 10** (0.5 s at 20 Hz). Window 1 → 10 improved RPM by 31 % (averaged over sensor setups).
4. **Reconstruction loss is essential**: L = L_BC + 0.5·(cube position) + 0.3·(clean joint pos) + 0.2·(clean joint vel).
   Without it the transformer is the worst student.
5. Student details: per-step obs (noisy joint pos/vel, prev action, prev joint command, goal command) linearly
   projected to 256-dim tokens (10 history + 1 action + 1 command), sinusoidal pos. embedding, 3 layers, 4 heads,
   FF 1024, [CLS] queries → action + reconstruction heads. 1200 epochs (6000 steps), batch 256, lr 1e-5, wd 0.01.
   Distillation is L2 to teacher actions (DAgger vs fixed dataset not stated).
6. **Motor-encoder gap is tendon stretch, friction and backlash** — none of it is in our sim. Joint deviations from
   commands encode cube size/weight (thumb deflects most). With motor-only proprio, candidate DR: per-episode
   joint offset, tendon-compliance proxy (lower/randomized stiffness), backlash/deadband on the action.
7. **Observation noise: add a per-episode Gaussian bias on joint positions** on top of per-step noise (we only have
   per-step). Paper doesn't give σ; start with ±0.02–0.05 rad (section 1).
8. **Confirmed choices we already have**: action-rate penalty ("critical" for smooth hardware motion), robot friction
   0.8–2.0, cube friction 0.3–1.0, cube mass 0.5–1.5. Their link mass scale is 0.95–1.05; check ours.
9. Wrist: fixed at 0 on hardware (no sensor); policy still outputs 17 joints → see fixed-wrist ablation.
10. Cube: they use 55 mm (also 65 mm); we have 45 mm → **confirm demo cube size**.
11. **Eval protocol to copy**: 3 × 60 s real trials; RPM, rotation accuracy, drop-free success rate, drop count.

## 8. Evaluation metrics (from the paper; use for every run, sim and real)

Reward is not comparable across variants (different obs noise / DR), so every run is scored with the paper's metrics on
a **fixed eval setting**: command 1.5 rad/s about z (fixed sign), 60 s episodes (1200 policy steps at 20 Hz), DR + obs
noise **on** (the training distribution of the baseline), same seed.

| Metric | Definition (ours, sim) | Paper (real, PT-Motor) |
|---|---|---|
| **RPM** | cube rotation about the hand z-axis in the commanded direction (unwrapped yaw / 2π) per minute, until the first drop | 9.33 |
| **Rotation accuracy (RA)** | % of time the cube's z angular velocity has the commanded sign (exact paper definition: check §IV) | 96.2 % |
| **Drop-free success rate (DFSR)** | % of 60 s trials without a drop | 98.1 % |
| **Drop count (DC)** | drops per 60 s trial (cube re-spawned after a drop, keep counting) | 0.33 |

- Sim: 3 seeds × 64 envs × 60 s per checkpoint. Real: 3 × 60 s trials, as in the paper.
- Implementation: `hpc/replay/record_rollout.py` already logs cube quat, angular velocity, command and dones per step;
  add `hpc/eval_metrics.py` that turns a recording into these four numbers (+ `--steps 1200`, fixed command, more envs).
- Also log RPM / DFSR at a lower command (0.9 rad/s) and with direction flips (deploy node's `goal_flip_period_s`).

## 9. Teacher → student plan (stages)

| Stage | What | Code | Status |
|---|---|---|---|
| 1 | Privileged teacher: PPO, actor + critic read the noise-free 81-dim `teacher` group; env also exposes the noisy 74-dim `policy` group for the student | task `orca-v2-inhand-repose-teacher-v0` (`OrcaV2ReposeTeacherEnvCfg`, `rsl_rl_teacher_student_cfg.py:PPORunnerCfgOrcaV2Teacher`) | written, smoke test |
| 2a | MLP student on a 10-step history of the noisy policy obs (Isaac Lab `history_length=10`) | `DistillationRunner` + `StudentTeacher` (rsl_rl built-in, DAgger-style: student acts, teacher labels); config only | TODO |
| 2b | **LSTM student** (paper: 7.0 RPM, 100 % DFSR with motor encoders) | `StudentTeacherRecurrent`, config only | TODO, preferred first student |
| 2c | Per-episode joint position bias in the student's obs noise | custom noise term (~30 lines) | TODO |
| 3 | Transformer student + reconstruction loss (paper's best, 9.3 RPM) | new module + `Distillation` subclass + recon obs group (~300 lines) | only if 2b looks good |
| 4 | Export | TorchScript wrapper that holds the history buffer / LSTM state inside `policy.pt` → still one 74-dim obs per call, **deploy node unchanged** | TODO |

Notes:
- The teacher is trained under full DR (sim only) so its labels cover the randomized dynamics the student must handle.
- Distillation loads the teacher checkpoint via `load_run` / `load_checkpoint` (train.py:170); `teacher_obs_normalization=True`
  so the PPO actor normalizer is reused.
- History must be initialized the same way at reset in sim (Isaac Lab fills the buffer with the first obs) and on deploy.

## 10. More ablation ideas from the paper

| Idea | Why | Effort |
|---|---|---|
| History window 1 / 3 / 6 / 10 | paper: RPM +31 % from 1 → 10; window 3 with motor encoders dropped RA to 74 % | config |
| Student MLP vs LSTM vs Transformer | MLP 5.1 / LSTM 7.0 / Transformer 9.3 RPM (motor encoders) | 2a/2b config, 3 code |
| Reconstruction loss on/off, weights (0.5 cube pos, 0.3 joint pos, 0.2 joint vel) | without it the transformer is the worst student (35 % DFSR) | with stage 3 |
| Cube size randomization (45/55/65 mm) | student infers size from joint deviations; robust to the real cube | small (scale DR or multi-asset) |
| Full SO(3) initial cube pose | paper samples it; ours? check `reset_object` | config |
| Link mass scale 0.95–1.05 | paper's range; check ours | config |
| Log-uniform joint stiffness/damping sampling | paper samples log-uniform; we use uniform scale | config (`distribution="log_uniform"`) |
| **Gravity on the hand links** | paper: largest reconstruction error in z "due to gravity-induced finger deflection" on hardware; our hand has `disable_gravity=True` | config, re-check PD gains |
| Exteroceptive teacher vs camera student (Subtask 2) | their Extero-PPO: 3.1 RPM but 100 % DFSR (slower but no drops) → cube pose helps stability, PT helps speed | later |

## 11. Literature review: other in-hand rotation work (subagent, 2026-10-27; summaries of paper text, [S] = snippet only)

| Paper | Takeaway for us |
|---|---|
| HORA 2210.04887 | 3-step joint pos + targets; RMA latent (8-d) from 30-step 1D-conv adaptation; real TTF 0.98 vs 0.72 plain DR. MLP with long raw history collapses (OOD rot 141 @3 steps → 36 @30) |
| RotateIt 2309.09979 | proprio-only student 79 vs 125 with shape; binary contacts add nothing |
| PenSpin 2407.18902 | sim-only transfer of dynamic skills fails (8–45 %); <50 real trajectories fine-tune → 55–81 % |
| AnyRotate 2405.07391 | palm-up z-rotation proprio-only; goal-increment keypoint reward beat angular-velocity reward; termination penalty 50 |
| Visual Dexterity 2211.11744 | servo hand: CMA-ES sysid of robot dynamics, then **small** robot DR; EMA 0.8; soft fingertips |
| Faive 2308.02453 | tendon hand: 5-step history, asym critic; reward from finite-differenced orientation (sim ang. vel exploited); measured joint pos ×0.5 (tendon stretch) |
| LEAP 2309.06440 | GRU + grasp cache; rotation reward clipped ±0.25 |
| DexNDM 2510.08556 | learned per-joint real dynamics → residual policy: 131 vs 19 rad real rotation |
| Aero Hand Open 2608.28578 | **closest analogue** (tendon, motor encoders only, cube z-rot zero-shot): reward ω_z − 100·fall − ‖Δa‖²; joint zero offsets ±0.05 rad, actuator gain ×0.8–1.2, joint dry friction ×0.5–2; sim servo 10–40 ms too fast |
| xHand 2601.02778 | randomized actuator model: PD, backlash deadband, torque-speed saturation |
| 2502.20396 | real2sim autotune from 4 min tracking data: best-MSE params 8/10 real vs worst 0/10; lag with prob 0.1 |
| Dactyl 1808.00177 [S] | each actuator 1-step delayed w.p. 0.5 per episode |

Ranked ideas taken into the run plan: (1) short history 3–5 steps for MLP (longer only behind conv/transformer);
(2) large drop penalty + clipped rotation reward; (3) grasp-cache resets; (4) random cube pushes (force ≈ 2·m,
p=0.25, decay 0.9); (5) actuator sysid + delay; (6) per-joint encoder offsets ±0.05 rad; (7) RMA latent as cheaper
alternative to the transformer student; (8) delta actions; (9) finite-difference angular velocity reward;
(10) runtime-tunable action scale / grasp offset on deploy. Our 45 mm cube is near HORA's failure size (<40 mm):
randomize scale ×0.9–1.1 and friction wider (0.3–3.0).

## Run plan / log

Variants are task ids (flags, no worktrees), defined in `tasks/orca_inhand/orca_v2_repose_ablation_env_cfg.py`.
Same seed (42), 8192 envs and PPO cfg as the baseline; compare B/C/sanity with A **at the same iteration** (A's
curve to 5k is in W&B). Submit with `train_job_ablation.sh` (TASK / RUN_NAME / ITERS env vars).

Results go to `ABLATION_RESULTS.md`: RPM / RA / DFSR / DC (section 8) at the final checkpoint, plus reward for reference.

| Run | Task id | Change vs baseline | Iters | Result |
|---|---|---|---|---|
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline (critic = policy obs) | 10k | reward ~493 (last 100 iters), drops ~62 % |
| B `abl_B_asym` | `orca-v2-inhand-repose-asym-v0` | critic: noise-free joints + cube pose (`TeacherObsGroupCfg`) | 5k | |
| C `abl_C_asym_dr` | `orca-v2-inhand-repose-asym-dr-v0` | B + cube lin/ang vel + sampled DR params (mass, friction, gain scales, gravity) | 5k | |
| sanity `sanity_nodr` | `orca-v2-inhand-repose-nodr-v0` | no physics/mount DR, no obs noise | 5k | expect higher sim reward than A (easier); not deployable |
| | | + action delay 0–2 steps | 5k | |
| TBD | | fixed 1-step action delay (sim) + next-tick actuation on deploy (section 6b) | 5k | |
| | | + compliant fingertip contacts (randomized stiffness) | 5k | |
| teacher | `orca-v2-inhand-repose-teacher-v0` | privileged teacher: `TeacherObsGroupCfg` as **actor + critic** input (sim only, not deployable) | 10k | |
| TBD student-LSTM | | distill teacher → LSTM student on noisy policy obs (section 9, stage 2b) | | |
| TBD student-PT | | distill teacher → transformer on 10-step motor-proprio history + recon loss (section 9, stage 3) | 6k steps | |
| TBD | | + per-episode joint position bias in obs noise | 5k | |
| wave 2 | `-delay-v0` | joint targets 1–2 steps late per episode | 5k | |
| wave 2 | `-jointbias-v0` | per-episode joint obs offset ±0.05 rad | 5k | |
| wave 2 | `-droppen-v0` | −100 return on drop | 5k | |
| wave 2 | `-hist3-v0` | 3-step obs history (MLP) | 5k | |
| wave 2 | `-handgrav-v0` | gravity on hand links | 5k | |
| wave 2 | `-wristfix-v0` | wrist not actuated (72-dim obs) | 5k | |
| TBD | | cube pushes, grasp cache, cube scale ×0.9–1.1, clipped rotation reward, finite-diff ω reward | 5k | |
| TBD | | fixed wrist: drop `right_wrist` from the action (`"(?!right_wrist).*"`, already commented out in `ActionsCfg`), hold it at 0 (stiff drive). Currently the policy actuates all 17 joints incl. the wrist. Matches the PT paper (arXiv 2605.21330: wrist unsensed, fixed at 0 on hardware) | 5k | |
