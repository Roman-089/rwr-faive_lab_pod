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
| 1 | Privileged teacher: PPO, actor + critic read the noise-free 81-dim `teacher` group; env also exposes the noisy 74-dim `policy` group for the student | task `orca-v2-inhand-repose-teacher-v0` (`OrcaV2ReposeTeacherEnvCfg`, `rsl_rl_teacher_student_cfg.py:PPORunnerCfgOrcaV2Teacher`) | queued (10k iters) |
| 2a | MLP student on a 10-step history of the noisy policy obs (Isaac Lab `history_length=10`) | task `-student-mlp-h10-v0` (`DistillationRunner` + `StudentTeacher`, DAgger-style: student acts, teacher labels) | written, needs teacher |
| 2b | **LSTM student** (paper: 7.0 RPM, 100 % DFSR with motor encoders) | task `-student-lstm-v0` (`StudentTeacherRecurrent`) | written, needs teacher |
| 2c | Per-episode joint position bias in the student's obs noise | custom noise term (~30 lines) | TODO |
| 3 | Transformer student + reconstruction loss (paper's best, 9.3 RPM) | task `-student-pt-v0`: `transformer_student.py` (CLS token over 10 per-step tokens, 3 layers / 4 heads / d 256 / FF 1024; recon targets = first 37 dims of the teacher obs) | written, CPU unit-tested, needs teacher |
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

## 12. Literature review: actuation, sensing, vision (subagent, 2026-10-27; PDFs read in full unless noted)

| Paper | Takeaway for us |
|---|---|
| SERNF 2602.09580 (ETH SRL, ORCA, **45 mm cube**) | distilled sim student "fragile – most trajectories fail before a full rotation"; needed 105 min real online RL → 6.25 RPM. No latency modelling in the teacher. Stiffness ×0.75–1.5, damping ×0.3–3.0 log-uniform. Vision: Mask R-CNN keypoints → PnP-RANSAC → Kalman, 16 Hz |
| ORCA 2504.04259 | no joint sensors; linear motor→joint map from auto-calibration; latency <0.2 s, slack adds latency |
| Faive 2308.02453 | joint readings ×0.5 at deploy improved real performance (tendon stretch); missing actuator sysid named as the main gap |
| Aero Hand 2608.28578 | sim actuator 10–40 ms faster than hardware (rise 90 vs 120 ms); open-loop replay of sim actions on hardware localised the gap |
| Hwangbo 1901.08652 | actuator net needs torque labels → **not possible for us** (no torque, no joint sensor) |
| Tan 1804.10332 | latency 15–19 ms; without actuator model or latency, controllers fail on hardware |
| OpenAI 1808.00177 | 1-step (80 ms) delay w.p. 0.5 per actuator; tendon backlash model; LSTM median 13 vs FF 3.5 |
| DeXtreme 2210.13702 | pose at 15 Hz locked to policy rate; ADR: pose refresh 1–6 steps, pose delay, action latency 0–1.5 steps, random pose injection p~U(0,0.3); stiffness ×0.3–3.5; smoothing only if no added latency |
| Visual Dexterity 2211.11744 | CMA-ES sysid of stiffness, damping **and velocity limit** from steps + sines |
| LEAP UHAS 2607.03570 | servo **velocity limits** had to be added and randomized |
| 2603.15789 | chirp + CMA-ES over friction, armature, **delay**: joint RMSE ~7° → <2° |
| 2502.20396 | sysid quality predicts real success (best-MSE 8/10, worst 0/10) |
| 2410.08650 | Stribeck + load-dependent friction beats Coulomb-viscous for Dynamixel |

Actionable, in order: (1) sim-in-loop actuator sysid from v2 chirps (Kp, Kd, armature, friction, delay, velocity
limit; tool exists in `scripts/motor_param_characterization/` (owner S1), needs v2 recordings at 20 Hz);
(2) delay at **physics-substep** granularity (0–6 substeps of 8.3 ms) on top of policy-step delay;
(3) servo velocity / slew limit on targets; (4) history encoder (see §9); (5) motor-side obs model
(q + τ/k_tendon, k randomized — untested idea), keep Faive's ×0.5 as a deploy knob; (6) sweep EMA α at deploy;
(7–9) Subtask 2: train with pose hold/delay/injection noise, run the estimator at a fixed integer rate with outlier
rejection instead of heavy smoothing, distil vision into a history student; (10) backlash model from chirp reversals.

## 13. Paper → run mapping and code availability (2026-10-28)

How the Phase 1 runs trace back to the literature, and which papers publish code we can port (checked 2026-10-28).

| Run / design choice | Source | Code |
|---|---|---|
| B asymmetric critic | Pinto et al. 2017 (1710.06542); also Faive, DeXtreme, Dactyl | – (standard) |
| C critic + DR params | RMA / HORA (DR params as privileged info), DeXtreme critic | – |
| hist3 vs hist10 (MLP) | HORA: MLP on long raw history collapses; Faive 5 steps; PT 10 steps | – |
| asym-lstm, student-lstm | Dactyl 1808.00177 (LSTM 13 vs FF 3.5); LEAP GRU | – |
| teacher → MLP-h10 / LSTM / transformer + recon | PT 2605.21330 | **not found** (arXiv lists none; an author's post mentions code + data, link unverified) |
| **rma-teacher / rma-student** (new) | HORA 2210.04887 | **yes, MIT**: [HaozhiQi/hora](https://github.com/HaozhiQi/hora). `ProprioAdaptTConv` ported 1:1 into `tasks/orca_inhand/rma.py` (30 steps, conv 9/2-5-5, 8-d tanh latent, MSE on latent) |
| **asym-droppen** (new) | AnyRotate 2405.07391 (termination 50), Aero Hand 2608.28578 (−100 fall) | AnyRotate: not found; Aero Hand: [TetherIA/aero-hand-open](https://github.com/TetherIA/aero-hand-open) has `sim_rl/` (MuJoCo Playground), rotation task unverified |
| entropy 5e-3 | legged_gym default 1e-2 vs our 1e-4 | – |
| metrics, perturbed eval | PT (metrics); Aero Hand, DeXtreme, Dactyl (delay / offset ranges) | – |
| next: grasp-cache resets | HORA, LEAP | **yes**: HORA `gen_grasp.py` + `cache/`; [leap-hand/LEAP_Hand_Sim](https://github.com/leap-hand/LEAP_Hand_Sim) (grasp cache per cube size) |
| later: PBT | DexPBT 2305.12127 | Isaac Lab PBT exists but **rl_games only** (we use rsl_rl) → would need porting |

HORA details that matter for us (from the code): reward values in the repo are higher than the paper because the reset
order follows legged_gym; their adaptation module's comment says 50 steps but the conv stack only fits 30.

## 14. Literature review: reward structures for in-hand rotation (subagent, 2026-10-28)

Our terms, from the code (`mdp/rewards.py`): direction 60 · cos(ω, ω*) (**independent of speed**); magnitude
30 · exp(−| |ω| − |ω*| | / 0.2) (Laplacian, not Gaussian); position 15 · exp(−d / 0.02 m); ang. acc. 1 · exp(−|Δω| / 0.5);
small joint-vel / action / action-rate penalties. ω = finite difference of the cube quaternion. Commands: |ω*| sampled in
60–100 % of 1.5 rad/s about ±z, resampled every 8 s (so the eval's 1.5 rad/s is the top of the training range).

| Work | Rotation reward | Stability / pose | Energy / smoothness | Drops | Curriculum | Source |
|---|---|---|---|---|---|---|
| HORA 2210.04887 | `clip(ω·k, ±0.5)` ×1 (speed up to a clip) | lin. vel −0.3‖v‖₁, pose −0.3 | torque −0.1‖τ‖², work −2.0 | reset, no penalty | – | code (public) |
| RotateIt / Vision+Touch 2309.09979 | same, + off-axis −0.1‖ω×k‖₁ | lin. vel −0.3, pose | work −2.0, torque −0.1 | reset | off-axis weight annealed 0 → −0.1 | paper |
| LEAP_Hand_Sim | `clip(ω_z, ±0.25)` ×1.25 | lin. vel −0.3, pose −0.1 | torque −0.1, work −1.0 | −10 per fallen step + reset | – | code |
| Aero Hand 2608.28578 / Playground `rotate_z` | unclipped 1·ω_z | none | action rate −1 | −100·dt on termination | – | code; only 0.51 rad/s in sim |
| ORCA PT 2605.21330 | **same as ours** (dir 60, mag 30, σ not given) | position 10 · exp(−d/0.02) | as ours | none | – | paper only; 11.8 RPM real |
| DLR Sievers 2022 2204.03698 | λ·Δθ, clipped ≈1 rad/s | −\|Δz\|, −\|Δxy\|, off-axis | action low-pass | reset | gravity 5→100 %, filter 0.12→0.5 s | paper |
| DLR 2024 2411.13148 | `min(Δθ, θ_clip)`, clip set by target speed | position progress, q⁴ | – | reset | horizon | paper: **"velocity-matching rewards failed"** |
| AnyRotate 2405.07391 | keypoint dist + 5·clip(Δθ·k, ±0.025/step) + bonus | pose −0.5, ω cap 0.6 | work −0.1, torque −0.05 | −50 | penalties scaled up with performance | paper |
| DeXtreme / IsaacLab reorient | 1/(\|rot err\|+0.1), +250 success | position −10·d | action −2e-4, rate −0.01 | 0 | ADR | code |

**Diagnosis (why we under-rotate, ~6 of 14 RPM):** per step, any slow rotation in the right direction earns ~75 of
~105 (direction 60 + position 15). At 0.63 rad/s the magnitude term pays 30·e^(−0.87/0.2) ≈ 0.4 with a gradient of
~2 per rad/s, while 1 cm of position error costs ~6. Speed is barely rewarded, translational wobble from fast gaiting
is punished hard, and a speed-independent survival bonus makes holding on worth more than rotating. Every other work
except ORCA PT rewards ω·k **linearly** (constant gradient) and clips it; DLR 2024 reports that velocity-matching
rewards failed and clipping at the target speed worked. Prediction: the queued B + −2000 drop penalty makes the policy
slower and safer, not faster.

**Reward ablations (on top of B, ranked):**
1. `asym-rotlin` — direction + magnitude → 90 · clip(ω·k, −0.5, |ω*|) / |ω*| and −5 · off-axis ‖ω⊥‖₁ (HORA/LEAP
   form, DLR 2024 clip, RotateIt off-axis). ~30× the speed gradient below the target, nothing for overspeed. Risk: drops.
2. `asym-rotsoft` — keep the forms, magnitude σ 0.2 → 0.75 and weight 30 → 60, direction 60 → 30. Risk: precision.
3. (next) position σ 0.02 → 0.05, weight 15 → 5, + object linear-velocity penalty −20·‖v‖₁ (HORA): let the cube wobble
   but not slide.
4. (next) rotlin + moderate drop penalty −200 (−10 after dt; LEAP −10/step, Aero −5).
5. (if rotlin is jerky) HORA work / torque penalties, annealed in over the first 30 % (RotateIt, AnyRotate).

Training rewards of these variants are not comparable with B's; compare by eval (RPM / RA / DFSR).

## 15. Next architecture candidates (agreed 2026-10-28, queue after the reward runs)

Base = the winner of the reward wave (B, `-rotlin` or `-rotsoft`, decided by eval). Single change each:

| Run | Change | Why |
|---|---|---|
| `alpha05` | EMA α 0.2 → 0.5 | α 0.2 at 20 Hz is a ~5-step (≈250 ms) time constant on joint targets; may cap gaiting speed |
| `relact` | relative joint targets (q_target = q_target,prev + s·a, HORA/LEAP) | small, fast finger motions are easy to express |
| `big` | actor/critic MLP 512-512-256-128 → 1024-1024-512-256 | capacity; SimBa (2410.09754) / BRO: larger normalized nets help PPO |
| student fine-tune | PPO fine-tuning of the best distilled student | close the gap to the teacher; known to be unstable at the start → check existing code / stabilizers first (critic warm-up, KL-to-teacher anneal) |

Order: decide after the reward evals; `alpha05` / `relact` change the action handling that the policy node must mirror
(allowed: we fork `rwr_faive_system`).

## Run plan / log

**Phase 1 (now): architecture + reward study** (decided 2026-10-28: find the best architecture and reward first, then iterate on sim2real; a fixed, enforced action delay matched in sim — section 6b, `DelayedEMAJointPositionToLimitsActionCfg` — is the first Phase 2 run).

**Architecture study.** Sim2real ablations (delay, joint bias, hand gravity, fixed wrist, ...) change
the training distribution and can only be judged on hardware, so they are registered but parked. Phase 1 compares
architectures that change what the policy can do in sim, all on the **baseline env (A)**:
MLP (A) · asymmetric critic (B) · critic + DR params (C) · MLP on 3-step history (hist3) · LSTM actor-critic (lstm) ·
privileged teacher → distilled students (MLP-h10, LSTM, transformer + recon). Each run is scored twice
(`ABLATION_RESULTS.md`): nominal eval, and **perturbed eval** = same policy with 50–100 ms action delay and ±0.05 rad
joint obs offset at test time only (`eval_policy.py --perturb`) as a sim proxy for robustness to unmodeled effects.

Variants are task ids (flags, no worktrees), defined in `tasks/orca_inhand/orca_v2_repose_ablation_env_cfg.py`.
Same seed (42), 8192 envs and PPO cfg as the baseline; compare B/C/sanity with A **at the same iteration** (A's
curve to 5k is in W&B). Submit with `train_job_ablation.sh` (TASK / RUN_NAME / ITERS env vars).

Results go to `ABLATION_RESULTS.md`: RPM / RA / DFSR / DC (section 8) at the final checkpoint, plus reward for reference.

| Run | Task id | Change vs baseline | Iters | Result |
|---|---|---|---|---|
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline (critic = policy obs) | 10k | reward ~493 (last 100 iters), drops ~62 % |
| B `abl_B_asym` | `orca-v2-inhand-repose-asym-v0` | critic: noise-free joints + cube pose (`TeacherObsGroupCfg`) | 5k | **5k (train, last 50 iters): reward 1573 vs A 477; drops 10 % vs A 65 %; ep. length 588 vs 374 / 640; action-rate penalty −0.76 vs A −0.28 (busier fingers, check smoothness).** Sim metrics (section 8) TBD |
| C `abl_C_asym_dr` | `orca-v2-inhand-repose-asym-dr-v0` | B + cube lin/ang vel + sampled DR params (mass, friction, gain scales, gravity) | 5k | 5k (train, iters 4900–4999): reward 1584 vs B 1568; drops 12 % vs B 10 %; ep. length 583. Ahead of B mid-run (1390 vs 1288 @2k) but ≈ B at the end — within single-seed noise, extra critic inputs not clearly worth it |
| sanity `sanity_nodr` | `orca-v2-inhand-repose-nodr-v0` | no physics/mount DR, no obs noise | 5k | 5k (train, iters 4900–4999): reward 1809, drops 10 %, ep. length 579 — symmetric critic learns fine without DR/noise, so A's weakness is critic + DR/noise; B recovers ~87 % of this upper bound. Not deployable |
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
