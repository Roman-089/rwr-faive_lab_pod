# Sim2real (Phase 2, parked)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

**Status: parked.** Sim2real ablations (delay, joint bias, hand gravity, fixed wrist, ...) change the training
distribution and can only be judged on hardware, so they are registered but parked during Phase 1. **Phase 2 starts with
a fixed, enforced action delay matched in sim** (see [Fixed action delay](#fixed-action-delay-sim--deploy),
`DelayedEMAJointPositionToLimitsActionCfg`). The sanity runs (no DR) are here too because they measure what DR costs.

## Question / hypothesis

What does the real hand have that the sim lacks, and which of these gaps matter? Close the biggest ones (latency,
encoder offsets, actuator dynamics, tendon effects) by modelling or enforcing them, fitted to measurements.

## What we do NOT model in training

- **No actuator noise / latency model.** Actions go straight to an ideal implicit PD drive every physics step:
  no command delay, no action noise, no backlash, no servo bandwidth limit beyond the PD gains themselves.
  Only the PD gains are randomized (stiffness ×0.75–1.5, damping ×0.3–3.0 per reset).
- **No realistic observation noise.** The only corruption is white Gaussian noise on `joint_pos` (std 0.10 on the
  limit-normalized value) and `joint_vel` (std 0.05). No observation delay, no per-episode encoder bias/offset,
  no quantization, no dropped packets.
- **No soft fingertips.** Fingers are rigid bodies with hard PhysX contacts; high friction (0.8–2.0) is the only
  stand-in for grippy pads (no contact patch / extra torsional grip).

The real hand will have all of these, so these are the first sim2real gaps to close.

## Deploy-repo facts (task PDF)

- Hardware: plain ORCA hand, **no joint sensors → proprio from motor encoders only** (= PT paper's *PT-Motor* setting).
- **Control frequency 20 Hz is fixed by the course**, in the simulation and on the real robot (physics 120 Hz,
  decimation 6; episode 32 s).
- The policy node `rwr_faive_system/experiments/rl_inference/policy_inference_cube_spin_v2.py` hard-codes the joint
  limits, the joint order, the observation order, the observation size **`OBS_DIM` (74)** and the scales; they must match
  the sim. Changing the obs means adding the same terms, in the same order, in the policy node and setting `OBS_DIM`.
- Each team uses one fork of `rwr_faive_lab` and of `rwr_faive_system` (branch main) → **we fork `rwr_faive_system`**
  and may change the policy node (needed for history students, `alpha05` / `relact`, next-tick actuation).
  `rwr_faive_system` was not released yet when these notes were written ("TBA").
- Chirp recorder: `experiments/orca/chirp.py` in `rwr_faive_system` (20 s, 0.1–5 Hz, ±10°, 20 Hz → 400 samples).

## Actuator and observation models

| Idea | Where | Values to try |
|---|---|---|
| Action delay (hold previous command) | action term / custom wrapper | 0, 1, 2 control steps (0 / 50 / 100 ms), sampled per episode |
| Action noise | `action_noise_model` in env cfg | Gaussian std 0.01–0.05 on joint targets |
| Observation delay | obs term with history buffer | 0–1 control steps, sampled per episode |
| Per-episode joint offset (calibration error) | `observation_noise_model` / custom | bias ±0.02–0.05 rad |
| Measure real latency + noise first | Subtask 1 chirps (task PDF §7) | fit the above ranges to the measurements |

From the PT paper ([literature](literature.md#orca-proprioceptive-transformer-paper-takeaways), takeaways 6, 7, 9, 10):
the motor-encoder gap is tendon stretch, friction and backlash (none of it in our sim); joint deviations from commands
encode cube size/weight (thumb deflects most). Candidate DR with motor-only proprio: per-episode joint offset,
tendon-compliance proxy (lower/randomized stiffness), backlash/deadband on the action. Add a per-episode Gaussian bias on
joint positions on top of per-step noise (paper gives no σ; start with ±0.02–0.05 rad). Wrist fixed at 0 on hardware.
Cube 55 mm (also 65 mm) in the paper vs our 45 mm → **confirm demo cube size**.

Actuation / sensing literature, ranked actions (delay at physics-substep granularity, servo velocity / slew limit,
sim-in-loop actuator sysid from v2 chirps, backlash from chirp reversals, motor-side obs model, Faive's ×0.5 deploy
knob): see [literature](literature.md#actuation-sensing-vision-literature-review). Aero Hand: joint zero offsets
±0.05 rad, actuator gain ×0.8–1.2, joint dry friction ×0.5–2; xHand: randomized actuator model (PD, backlash deadband,
torque-speed saturation); Dactyl: each actuator 1-step delayed w.p. 0.5 per episode.

## Domain randomization ranges

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

More DR ideas (PT paper and the [in-hand rotation review](literature.md#in-hand-rotation-literature-review)):

| Idea | Why | Effort |
|---|---|---|
| Cube size randomization (45/55/65 mm) | student infers size from joint deviations; robust to the real cube | small (scale DR or multi-asset) |
| Full SO(3) initial cube pose | paper samples it; ours? check `reset_object` | config |
| Link mass scale 0.95–1.05 | paper's range; check ours | config |
| Log-uniform joint stiffness/damping sampling | paper samples log-uniform; we use uniform scale | config (`distribution="log_uniform"`) |
| **Gravity on the hand links** | paper: largest reconstruction error in z "due to gravity-induced finger deflection" on hardware; our hand has `disable_gravity=True` | config, re-check PD gains |
| Exteroceptive teacher vs camera student (Subtask 2) | their Extero-PPO: 3.1 RPM but 100 % DFSR (slower but no drops) → cube pose helps stability, PT helps speed | later |
| Cube scale ×0.9–1.1, friction wider (0.3–3.0) | our 45 mm cube is near HORA's failure size (<40 mm) | config |
| Random cube pushes (force ≈ 2·m, p=0.25, decay 0.9) | recovery behaviour | config |

## Fixed action delay (sim + deploy)

Deploy code can only **add** delay, never remove the inherent one (inference, ROS hop, servo bus, servo's internal loop).
So make the total delay **fixed and known** and train for exactly that (the latency-enforcement idea):

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

## Runs

### Sanity runs (no DR)

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Ep. len | Drop % |
|---|---|---|---|---|---|---|---|---|
| sanity `sanity_nodr` | `-nodr-v0` | no physics/mount DR, no obs noise (not deployable) | 29829724 / A10 | 5k | done (2026-10-27 22:45) | ~1793 | ~574 / 640 | ~10 % |
| `sanity_nodr_on_dr` | eval only | sanity `model_4999` evaluated on the DR task | 29836428 | – | queued | | | |

Run-plan summary: 5k (train, iters 4900–4999): reward 1809, drops 10 %, ep. length 579 — symmetric critic learns fine
without DR/noise, so A's weakness is critic + DR/noise; B recovers ~87 % of this upper bound.

Sanity eval (on its own no-DR task, nominal / perturbed):

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 8.92 / 9.09 | 76.6 / 75.1 | 86 / 64 | 0.14 / 0.55 |
| 2000 | 9.32 / 9.41 | 77.8 / 78.0 | 72 / 67 | 0.34 / 0.52 |
| 3000 | 9.85 / 9.25 | 82.6 / 76.6 | 89 / 66 | 0.11 / 0.53 |
| 4000 | 9.43 / 9.62 | 80.2 / 77.5 | 88 / 53 | 0.14 / 0.75 |
| 4999 | **10.55 / 10.59** | **85.8 / 85.7** | 83 / 69 | 0.19 / 0.47 |

### Registered / planned sim2real runs (parked)

Task ids from `orca_v2_repose_ablation_env_cfg.py`, single change vs the baseline A.

| Run | Task id | Change vs baseline | Iters | Status |
|---|---|---|---|---|
| wave 2 | `-delay-v0` | joint targets 1–2 steps late per episode (50–100 ms, 6–12 substeps) | 5k | registered, parked |
| wave 2 | `-jointbias-v0` | per-episode joint obs offset ±0.05 rad | 5k | registered, parked |
| wave 2 | `-handgrav-v0` | gravity on hand links | 5k | registered, parked |
| wave 2 | `-wristfix-v0` | wrist not actuated (72-dim obs) | 5k | registered, parked |
| fixed-delay run (first Phase 2 run) | TBD | fixed 1-step action delay (sim) + next-tick actuation on deploy (`DelayedEMAJointPositionToLimitsActionCfg`) | 5k | planned |
| TBD | | + action delay 0–2 steps | 5k | TBD |
| TBD | | + compliant fingertip contacts (randomized stiffness) | 5k | TBD |
| TBD | | + per-episode joint position bias in obs noise | 5k | TBD |
| TBD | | cube pushes, grasp cache, cube scale ×0.9–1.1 | 5k | TBD |
| TBD | | fixed wrist: drop `right_wrist` from the action (`"(?!right_wrist).*"`, already commented out in `ActionsCfg`), hold it at 0 (stiff drive). Currently the policy actuates all 17 joints incl. the wrist. Matches the PT paper (arXiv 2605.21330: wrist unsensed, fixed at 0 on hardware) | 5k | TBD |

(`-droppen-v0` and `-hist3-v0` were also in wave 2; see [rewards](rewards.md) and [history](history_and_memory.md).)

## Log

- **2026-10-28 04:00 — sanity eval** (no DR / no noise, evaluated on its own no-DR task): table above.
  |cmd| 1.5 rad/s = 14.3 RPM, so the policy tracks ~74 % of the commanded speed; the perturbation (delay + joint
  offset) does not slow it but roughly triples drops. ~8× A's best RPM — but **not comparable to A** (evaluated without
  DR / noise). It is the upper reference for what the reward / action space allow; the gap to it is what DR + obs
  noise cost. To separate "DR makes the task harder" from "DR makes learning harder", the sanity `model_4999` is also
  evaluated on the DR task (`sanity_nodr_on_dr`, job 29836428).
- **2026-10-28 06:45** — gap to the no-DR reference is ~2× in RPM: DR + obs noise cost half the speed
  ([asymmetric critic](asymmetric_critic.md#log)).
- **2026-10-28 07:35** — RMA stage 1 shows that knowing the DR sample buys little; the remaining gap to the no-DR
  reference (10.6 RPM) is more likely obs noise and the harder dynamics themselves ([RMA](rma.md)).

## Conclusions so far

- DR + obs noise cost about half the speed (10.55 vs ~5.5–6.5 RPM); not mainly a system-identification problem.
- The perturbed eval (delay + offset) roughly triples drops for every policy, but does not slow it down.

## Next steps / open questions

- `sanity_nodr_on_dr` result.
- Phase 2: fixed-delay run first; measure the inherent delay from chirps; confirm the demo cube size; fork
  `rwr_faive_system` once released.
