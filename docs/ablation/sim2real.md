# Sim2real (Phase 2, parked)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

**Status: parked.** Sim2real ablations (delay, joint bias, hand gravity, fixed wrist, ...) change the training
distribution and can only be judged on hardware, so they are registered but parked during Phase 1. **Phase 2 starts with
a fixed, enforced action delay matched in sim** (see [Fixed action delay](#fixed-action-delay-sim--deploy),
`DelayedEMAJointPositionToLimitsActionCfg`). The sanity runs (no DR) are here too because they measure what DR costs.

## Phase 2 checklist (reminder: do these once sim2real starts)

The user parked sim2real on 2026-10-28 and asked to be reminded of these experiments when Phase 2 starts:

1. **Fixed action delay**, matched in sim and enforced on deploy: the agreed first Phase 2 run
   ([below](#fixed-action-delay-sim--deploy)).
2. **Random action holding** (`-asym-acthold-v0`, [below](#random-action-holding)): implemented + CPU-tested, not run.
   Combine with the delay run.
3. **Actuator network** after arXiv 2603.04351 / Hwangbo 1901.08652: [actuator_net.md](actuator_net.md). Retrain on S1's
   new ORCA v2 chirps first; the existing chirps are from the v1b hand.
4. **Sim2sim in MuJoCo before the hardware** ([sim2sim.md](sim2sim.md), harness being built 2026-10-29): run the exported
   policy on CPU in MuJoCo with the ORCA v2 MJCF (`orcahand/orcahand_description`) as a stand-in for the real hand.
   Step 1 (now): MuJoCo task + obs builder identical to Isaac's 74-dim observation, scored with our eval metrics; checks
   that Isaac and MuJoCo move a joint the same way for the same command. Step 2 (when `rwr_faive_system` is released):
   wrap MuJoCo as a fake ROS hand driver so the course's policy node runs unchanged end to end (joint order, limits, obs
   scales, EMA, 20 Hz timing, `policy_history.pt` + `reset()`). Catches interface bugs and PhysX-specific exploits;
   MuJoCo is not reality (no backlash / delay unless the actuator-net findings are added).

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

- **Joint order of the policy** (found by the sim2sim work, [sim2sim](sim2sim.md#findings-for-the-policy-node--deploy)):
  Isaac's articulation order (wrist, the four abd, thumb_cmc, the four mcp, thumb_abd, the four pip, thumb_mcp,
  thumb_dip), **not** the order listed in `orca_v2.py`; the export JSON does not record it yet. The policy node must
  map joints in this order.
- **`last_action` must be the raw policy output**, not clipped: 58 % of rotlin's actions lie outside [−1, 1].

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
| Initial joint positions | U(0.2·lower, 0.2·upper) around the default pose (`reset_joints_within_limits_range`, operation `scale`; corrected 2026-10-29, the earlier note "effectively none" was wrong) | keep; match how the cube is placed on the real hand |
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

## Random action holding

**Status: implemented + CPU-tested, NOT run.** Queued for Phase 2, to be combined with the fixed action-delay run
(below). Task `orca-v2-inhand-repose-asym-acthold-v0` (`OrcaV2ReposeAsymActHoldEnvCfg`, on top of B); action term
`HoldingEMAJointPositionToLimitsActionCfg` in `tasks/orca_inhand/actions_ablation.py`. Deploy: unchanged.

**Model.** On the real hand the policy node computes the EMA target every 50 ms and sends it to the servos over the bus.
If the packet is lost, corrupted or late, the servos keep the previous target, and the node does not know. In sim, each
env samples p ~ U(0, `hold_prob_max`) per episode. Each policy step, with probability p, the target applied to the
joints (for all 6 substeps) stays the previously applied one. The command the policy sees is the one the node sent, as on
deploy: the EMA state `_prev_applied_actions` (= `last_joint_pos_cmd`) and `last_action` update as usual. Only
`processed_actions`, which `apply_actions` writes to the joints, is held. The policy can notice a hold only through
`joint_pos` / `joint_vel`. After a hold, the next applied target is the current EMA state. On reset, the held target is
the reset joint position (= the parent's reset EMA state). `hold_prob_max = 0` runs the parent's code path without
random draws, so it reproduces the parent bit for bit (tested).

| Knob | Default | Meaning / why |
|---|---|---|
| `hold_prob_max` | 0.1 | p ~ U(0, 0.1) per env and episode: on average 5 % of steps are stale, at most 10 %, and some episodes are almost clean. Same order as TeleDexter's stale joint-steps (our estimate below). Re-fit once real loss / late rates are logged on deploy |
| `hold_steps_range` | (1, 1) | hold length in policy steps, per event. A lost or late packet costs one tick, then the next command overwrites it. Longer holds (node stalls, TeleDexter's 1–10-step freezes) are a knob, e.g. (1, 3) |
| `per_joint` | False | all 17 joints held together, because one sync-write packet carries every servo's target. `True` holds each joint independently (per-servo writes / TeleDexter's desynchronized joints) |

With EMA α 0.2 the target moves only 20 % of the way per step, so a single 50 ms hold is a mild perturbation. If the
run shows no effect on the perturbed eval or on hardware, try bursts and `per_joint` next.

**Paper basis.** TeleDexter (Li et al., "Towards Human-level Dexterous Teleoperation",
[arXiv 2607.11481](https://arxiv.org/abs/2607.11481), §3.1, App. C.3 / C.7 / C.8, Tab. 5;
[project page](https://bigai-dex.github.io/blog/teledexter/)) calls it *random action masking*:
ã_t = m_t ⊙ a_t + (1 − m_t) ⊙ ã_{t−1}, so masked DoFs keep their previous command.
- Mechanism (C.8): at each env step, if no mask is active, a new one starts with p_mask = 0.15. It freezes
  n_m = 3 DoFs (drawn uniformly without replacement, of 16 on LEAP / 22 on SharpaWave) for d ~ U{1, d_max} control steps.
  d_max is ramped from 1 to 10 by their curriculum (C.3, done within the first 2·10⁹ env steps). Masks refresh
  asynchronously across ~62K envs. From these numbers (our estimate, not in the paper), ~2–3 % of joint-steps are stale
  at the start of the curriculum and ~6–9 % at the end.
- Motivation: the policy should not overfit to "the perfectly synchronized actuation of simulation". The real failure
  modes they name are "motor lag, backlash, occasional missed commands on the SharpaWave SDK".
- Evidence is real-world only (Tab. 5, success rate / tracking %, with vs without masking): HammerUse 66.7 / 86.7 vs
  33.3 / 57.1, ScrewdriverUse 73.3 / 86.7 vs 0.0 / 36.0, CuboidReorient 80.0 / 86.7 vs 26.7 / 51.1. They call it "the
  single most impactful sim-to-real intervention we tested". It is combined with their DR (C.7), which includes
  observation latency (2-frame queue, sampled w.p. 0.5) and joint position noise σ 0.1.
- Not stated in the paper: the sim control rate (60 Hz reference motion, 30 Hz teleoperation). Also not stated: whether
  the observed "previous low-level command" ã_{t−1} is the masked or the unmasked command (the notation suggests
  masked), and how masking interacts with their residual action (a^cmd_t = clip(a^cmd_{t−1} + 0.1·deadzone(a_t)), C.1).
  There is no sim ablation and no sweep of p_mask / n_m.

**Differences from TeleDexter, and why.**
1. Whole hand by default instead of 3 joints, because our failure model is a lost bus packet.
2. The observation is the command the node sent, not the held one, because the deployed node cannot know about a lost
   packet. If TeleDexter's obs is the masked command, it tells the sim policy which joints are stale; the real policy
   never has that information.
3. No curriculum: holds are single steps, and the per-episode U(0, p_max) already mixes clean and perturbed episodes.

Related precedent: Dactyl delays each actuator by 1 step with probability 0.5 per episode (see above).

**Phase 2 plan.** Combine holding with the fixed action delay:
`class _(DelayedEMAJointPositionToLimitsAction, HoldingEMAJointPositionToLimitsAction)`, plus a cfg inheriting both cfgs.
The MRO is Delayed → Holding → EMA, so the held target enters the delay FIFO. This is CPU-tested, but no task id is
registered yet. Run the fixed-delay task vs fixed-delay + holding with the same seed. To fit `hold_prob_max`, log in the
policy node how often a command misses its tick, and count servo-write errors.

**Eval caveat.** `scripts/eval/eval_policy.py` evaluates a task with its own action term, so the nominal eval of an
`-acthold` checkpoint runs *with* holding (harder than every other run's nominal). `--perturb` replaces the action term
with the delay term, which drops the holding. For comparable numbers, evaluate `-acthold` checkpoints on `-asym-v0`
(same observations and critic, only the action term differs), and on `-asym-acthold-v0` for robustness to holds.

**Verification (CPU, isaaclab stubbed).** Test:
`/tmp/claude-322334/-zhome-b3-e-222334/e98b2254-0dd6-46ce-be5a-af9ce9954bee/scratchpad/holdtest/test_acthold.py`.
It runs against the **real** Isaac Lab `joint_actions_to_limits.py` (the parent) and the real `configclass`, with a
stand-in articulation that tracks its targets with a first-order lag. Results:
- p_max 0 is bit-identical to `EMAJointPositionToLimitsAction` (60 steps × 6 substeps, partial resets) and consumes no
  random numbers.
- p = 1 keeps the applied target at the reset pose on every substep, while `_prev_applied_actions` equals the parent's
  EMA state.
- p = 1 / p = 0 envs side by side are independent. Joints are held together, or independently with `per_joint`.
- The hold rate is 0.0501 for mean p 0.0502 (4096 envs × 200 steps; per-env rate vs p: slope 1.00). With fixed p, the
  rate is 0.2999 for 0.3 and 0.0497 for 0.05. Bursts U{1..3} at p 0.1 give 0.180 (theory 0.182).
- Reset sets the EMA state and the held target to the joint position, clears the counters, resamples p in [0, 0.1] and
  leaves the other envs untouched.
- Composition with the delay term works as described above.
- Three deliberate bugs (no reset of the held target, EMA held too, RNG draws at p 0) are each caught.

Not verified: env construction in Isaac Sim (the env cfg is checked statically only) and training behaviour.

## Runs

### Sanity runs (no DR)

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Ep. len | Drop % |
|---|---|---|---|---|---|---|---|---|
| sanity `sanity_nodr` | `-nodr-v0` | no physics/mount DR, no obs noise (not deployable) | 29829724 / A10 | 5k | done (2026-10-27 22:45) | ~1793 | ~574 / 640 | ~10 % |
| `sanity_nodr_on_dr` | eval only | sanity `model_4999` evaluated on the DR task | 29836428 / A10 | – | done (2026-10-28 17:05): **3.32 / 4.97 RPM, RA 35.6 / 43.2 %, DFSR 55 / 42 %, DC 0.86 / 1.48** (nominal / perturbed) | | | |
| `ent5e-3_on_nodr` | eval only | `abl_B_asym_ent5e-3` `model_4999` evaluated without DR (`-asym-nodr-v0`, new eval variant) | 29842299 / A10 | – | done (2026-10-29 08:00): **8.58 / 8.07 RPM, RA 87.8 / 79.7 %, DFSR 95 / 84 %, DC 0.05 / 0.23** (nominal / perturbed) | | | |

What DR costs (nominal RPM): trained without DR, tested without DR 10.55 → trained without DR, tested with DR 3.32
(collapses); trained with DR (`ent5e-3`), tested with DR 7.33 → the same policy tested without DR 8.58 (+17 %, and
95 % drop-free). So the DR-trained policy is only modestly slowed by the randomized physics at test time; most of the
gap to the no-DR policy (8.58 vs 10.55) is the more conservative gait it learned. Narrowing DR to the measured hand
(Phase 2) can recover part of that ~2 RPM, not more.

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
| Phase 2, with the fixed delay | `-asym-acthold-v0` | **on B**: [random action holding](#random-action-holding): joint targets not updated w.p. p ~ U(0, 0.1) per step | 5k | implemented + CPU-tested; Phase 2 |
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
- **DR in training is necessary** (2026-10-28 17:05): the no-DR policy collapses on the DR task (10.55 → 3.32 RPM,
  RA 86 → 36 %, drops 0.19 → 0.86 per trial), clearly below every DR-trained policy there (B 5.45, ent5e-3 7.33). The
  mirror eval (best DR-trained policy on the no-DR task, queued) tells how much DR costs at test time vs in learning.
- The perturbed eval (delay + offset) roughly triples drops for every policy, but does not slow it down.

## Next steps / open questions

- `ent5e-3_on_nodr` result (mirror of `sanity_nodr_on_dr`).
- Phase 2: fixed-delay run first; measure the inherent delay from chirps; confirm the demo cube size; fork
  `rwr_faive_system` once released.
- Phase 2: fixed delay + [random action holding](#random-action-holding) (implemented, not run); log the real
  missed-command rate in the policy node to fit `hold_prob_max`.

## ADR on action delay and encoder offset (2026-10-30)

User request (2026-10-30 21:40): make the cube-position ×3 policy (`abl_B_onedir_droppen_pos45_ent5e-3`: 12.8 RPM,
88 % drop-free, perturbed 49 %, no tumbling) more robust for deployment, including action delay, following OpenAI's
automatic domain randomization (ADR, arXiv 1910.07113). This unparks the delay / offset part of the Phase 2 list.

`adr.py` (ours; subclasses the S0 terms in `mdp/sim2real.py`): one global level L ∈ [0, 1] scales the two gaps the
perturbed eval tests and the env does not randomize yet (it already randomizes gains, friction, masses, gravity, joint
limits):
- action delay per episode U[0, round(L·15)] physics substeps (≤ 125 ms; the eval uses 6–12 = 50–100 ms);
- joint-encoder offset per episode and joint U(±L·0.08 rad) (eval ±0.05).

`adr_level` (curriculum term, logged as `Curriculum/adr`): every 480 env steps (20 iterations) and ≥ 1000 ended episodes,
L += 0.05 if fewer than 25 % of the ended episodes were drops, L −= 0.05 above 40 %. New episodes sample with the
current L. Unit test on a stub env: 0 → 0.30 while the drop share is 10 %, back down at 60 %. Critic obs unchanged, so
the position ×3 checkpoint resumes directly: `abl_B_onedir_droppen_pos45_adr_from5e-3` (+3000 iterations, default
entropy, task `-asym-onedir-droppen-adr-v0`). Judged on the perturbed eval (drop-free rate, DC) at an acceptable cost in
nominal RPM, and on `Curriculum/adr` reaching ≥ 0.8 (the eval's ranges sit at L ≈ 0.6–0.8).

**Result (2026-10-31 02:50, pod, W&B on6yv8hs).** `Curriculum/adr` reached 1.00 (delay U[0, 15] substeps, offset
±0.08 rad) and held there (drop share 22–26 %, between the 25 / 40 % thresholds). Isaac, +z, nominal / perturbed
(the perturbed eval swaps in the fixed 6–12 substep delay and ±0.05 rad offset regardless of the training task; the
nominal eval runs at L = 0):

| checkpoint | 5500 | 6000 | 6500 | 7000 | 7998 | last-3 mean | before ADR (pos45, last-3) |
|---|---|---|---|---|---|---|---|
| RPM | 12.2 / 15.5 | 12.4 / 15.0 | 12.4 / 15.3 | 12.3 / 15.8 | 12.6 / 15.8 | 12.4 / 15.6 | 12.8 / 13.6 |
| DFSR % | 89 / 75 | 91 / 69 | 91 / 78 | **95 / 84** | 88 / 78 | **91 / 80** | 88 / 49 |
| DC | 0.11 / 0.42 | 0.09 / 0.38 | 0.09 / 0.30 | 0.05 / 0.19 | 0.14 / 0.31 | 0.09 / 0.27 | 0.15 / 1.07 |

ADR lifts the perturbed drop-free rate from 49 to **80 %** (drops per trial 1.07 → 0.27) at no nominal cost; the
best result under the perturbed eval so far by a wide margin. Under 50–100 ms delay it even turns faster (15.6 vs
12.4 RPM): trained mostly with delay (mean 7.5 substeps at L = 1), its gait is tuned for late targets, so on a
delay-free sim it is slightly conservative. Cube motion (model_7998, nominal): tilt 16.5° / 23.5° p90, 3.5 face
changes / min, wrist 0.34 rad / 0.52 rad/s, as flat as before. model_7000 (95 / 84 %) exported bit-exact:
`logs/rsl_rl/orca_v2_repose/2026-10-30_22-18-44_abl_B_onedir_droppen_pos45_adr_from5e-3_r6k/exported/policy_history.pt`
(params from the resumed run, same policy obs; the ADR run's own config in `params/wandb_config_on6yv8hs.json`).
MuJoCo check: job 29858891. Same recipe on the tilt-penalty policy: `upright_adr` (pod, W&B 6rsbkqqm, since 02:41Z).

**Second ADR fine-tune: the tilt-penalty policy (2026-10-31 05:40, pod, W&B 6rsbkqqm).** `-asym-onedir-droppen-upright-adr-v0`,
resume of `abl_B_onedir_droppen_upright_ent1e-4_r6k` model_4999, +2000 iterations; `Curriculum/adr` 1.00 from ~1100
iterations in, drop share ~22 %. Isaac, +z, nominal / perturbed:

| checkpoint | 5000 | 5500 | 6000 | 6500 | 6998 | last-3 mean | before ADR (last-3) |
|---|---|---|---|---|---|---|---|
| RPM | 11.2 / 10.3 | 10.6 / 13.2 | 11.4 / 13.3 | 11.3 / 13.1 | 11.6 / 13.5 | 11.4 / 13.3 | 11.4 / 11.0 |
| DFSR % | 91 / 48 | **94 / 80** | 91 / 77 | 91 / 70 | 91 / **83** | **91 / 77** | 93 / 54 |
| DC | 0.12 / 1.05 | 0.06 / 0.28 | 0.09 / 0.25 | 0.11 / 0.42 | 0.12 / 0.25 | 0.11 / 0.31 | 0.07 / 0.83 |

Same picture as on position ×3: perturbed drop-free +23 points, drops under delay / offset −63 %, nominal unchanged, and
faster under delay. The cube stays as flat as before ADR (model_6998: tilt 7.4° / 14.7° p90, 2.5 face changes / min) and
the wrist as calm (0.28 rad / 0.08 rad/s). **Best deployable policy so far** (flat + calm + robust); the final model_6998 (91 / 83 %, pushed by the pod: only
the save-interval checkpoints are on W&B) exported bit-exact as `logs/rsl_rl/orca_v2_repose/2026-10-31_02-41-27_abl_B_onedir_droppen_upright_adr_r6k/
exported/policy_history.pt`; model_5500 (94 / 80 %) in `exported_model5500/`. MuJoCo: job 29859208.
Caveat: the base policy is seed 0 of the tilt penalty, the good draw (seed 1: 9.6 RPM / 74 %).

**Third ADR fine-tune: tilt + position ×3 (2026-10-31 09:20, pod, W&B s1jru0ly).** Resume of
`abl_B_onedir_droppen_upright_pos45_ent1e-4_r6k` model_3999, +2000 iterations, `-asym-onedir-droppen-upright-adr-v0` +
position ×3. Isaac, +z, nominal / perturbed:

| checkpoint | 4000 | 4500 | 5000 | 5500 | 5998 | last-3 mean | before ADR (last-3) |
|---|---|---|---|---|---|---|---|
| RPM | 14.1 / 12.9 | 14.1 / 14.7 | 14.3 / 15.3 | 14.2 / 16.0 | 13.9 / 15.4 | **14.1 / 15.6** | 14.1 / 11.9 |
| DFSR % | 95 / 59 | 92 / 77 | 91 / 77 | **98** / 72 | 86 / 72 | **92 / 73** | 92 / 50 |
| DC | 0.06 / 0.61 | 0.08 / 0.38 | 0.11 / 0.33 | 0.02 / 0.36 | 0.14 / 0.39 | 0.09 / 0.36 | 0.06 / 1.0 |

Third replicate of the ADR effect: perturbed drop-free +23 points, nominal unchanged, faster under delay. The fastest
robust policy so far (+2.5 RPM over tilt penalty + ADR at a similar perturbed rate). model_5500 (98 / 72 %) exported
bit-exact (W&B checkpoint; params from the resumed run, `params/SOURCE.txt`). MuJoCo: job 29859924.

**Fourth ADR fine-tune: the weaker tilt-penalty seed (2026-10-31 14:40, DTU, W&B 2kv2feur).** Resume of
`abl_B_onedir_droppen_upright_ent1e-4_seed1` model_4999 (9.6 RPM, 74 / 39 %), +2000 it, `-upright-adr-v0`. Last-3 mean:
nominal 8.66 RPM / 80.7 % (+6 points), perturbed 9.79 RPM / 52.6 % (+14 points), cube flat (7.8°), wrist 0.08 rad/s;
`--servo` 5.57 / 25.5 %. ADR helps every base, but less on a weak one (+14 points vs +23–31 for position ×3, tilt seed
0 and tilt + position ×3).

## Deadline plan: deployable policies by 2026-10-31 10:00 Zurich (09:00Z)

User request 2026-10-30 22:30: "robust, deployable policies by 10 am". Teacher → student needs ~12 h (6k + 2k on the
pod) and cannot make it, so the deliverables are directly deployable +z PPO policies that keep the cube flat, plus the
ADR fine-tunes, exported with `export_history_policy.py` (bit-exact self-check) and checked in MuJoCo (informational):

| Candidate | Status | ETA (UTC) |
|---|---|---|
| A `student_pt_rotlin` (p3dxn5pm), bidirectional fallback | exported | done |
| B `abl_B_onedir_droppen_pos45_ent5e-3`, +z, flat (16°), 88 % / perturbed 49 % | exported 2026-10-30 22:40; MuJoCo job queued | done |
| B-ADR `abl_B_onedir_droppen_pos45_adr_from5e-3_r6k`: 12.4 RPM, 91 % / **perturbed 80 %**, flat (16°) | exported 02:55Z (model_7000) | done |
| C `abl_B_onedir_droppen_upright_ent1e-4_r6k` (tilt penalty): 11.4 RPM, 93 % / perturbed 54 %, flat (7°), calmest wrist; MuJoCo 7.2 RPM (best) | exported 01:40Z (model_4000) | done |
| **C-ADR** upright + ADR fine-tune (W&B 6rsbkqqm): 11.4 RPM, 91 % / **perturbed 77 %**, flat (7°), calm wrist | exported 06:45Z (model_6998; model_5500 as alternative) | done |
| `abl_B_onedir_droppen_pos45_ent1e-4_r6k`, `..._upright_pos45_ent1e-4_r6k` (pod, 4000 it) | queued (00:40Z slots) | ~07:00 |
| teacher with the winning reward → transformer student (afternoon upgrade) | config ready (`-teacher-onedir-droppen-upright-v0` ± position ×3) | ~13:00–15:00 |

Pod plan for the slots freeing 05:50–09:30Z (sent 04:10Z): tilt + smoothness ×10 (4000 it, DTU copy cancelled when
it starts); if tilt + position ×3 passes the flat / drop-free rule, its ADR fine-tune (resume model_3999, +2000 it), else
tilt + fixed wrist; tilt + fixed wrist or ADR from scratch; after the teacher (ETA ~09:05Z) the transformer student
`student_pt_onedir_droppen_upright_pos45` (2000 it).

Deployment notes for every +z PPO candidate: command +z at 1.5 rad/s (the policy ignores the magnitude), 20 Hz,
17-joint position targets through EMA α 0.2 to the joint limits, single-step 74-dim observation (normalization inside
`exported/policy_history.pt`).

Decision rule handed to the pod worker (2026-10-30 22:55, so the night slots never wait): when the tilt-penalty run
`abl_B_onedir_droppen_upright_ent1e-4_r6k` is evaluated, it is GOOD if `cube_motion.py` gives tilt < 15° and < 20 face
changes / min and model_4999 is ≥ 60 % drop-free. GOOD → teacher `-teacher-onedir-droppen-upright-v0` + position ×3
(6k) into its slot, then the upright ADR fine-tune (2k, resume from its model_4999) into the ~04:00Z slot. Not GOOD →
teacher `-teacher-onedir-droppen-v0` + position ×3, and the held single-knob runs continue.

ADR fine-tune started on the pod 2026-10-30 22:18Z (W&B on6yv8hs): resumed at 4999/7999, ADR action term (delay 0–15), ADR offset (0.08 rad), curriculum active, entropy 1e-4, position ×3 — all checked by the worker.

## First real-hand results (2026-10-31, interim, from the deploy session `deploy-rwr`)

Run on the real ORCA v2 with the unchanged course node: `orca_v2_zrot_deploy` #1–#3, `deployable_policies` #4 and the
bidirectional transformer student (4 trials). The hand lay flat with a −30° wrist offset in a relay, so the palm's angle
to gravity matched the training tilt. Measured (full report with CSVs to follow as a W&B dataset):

- **Latency:** command → joint starts moving 60–75 ms (fingers), ~90 ms (wrist); median 69 ms with ROS (+0–7 ms).
  Effective lag below 1 Hz: 80–93 ms fingers, **147 ms wrist**. ADR's delay range (0–125 ms) covers the fingers, not the
  wrist.
- **Servos are slew limited:** peak 120–180°/s on 15° steps, acceleration limit 22.8 rad/s² (motor side, orca_core
  default profile), bandwidth 2.3–3 Hz fingers, 1.6 Hz wrist. **The sim has no velocity or acceleration limit**: the PD
  tracks the EMA target, which moves by up to α·range = 0.2 × 125° = 25° per 50 ms (~500°/s) for MCP / PIP.
- In the trials the policies saturate their raw actions 50–99 % of the steps and command 60–120° swings; the real joints
  follow 150–300 ms late with 13–52° RMS tracking error.
- Joint-position noise at rest ~0 (training adds 0.1 normalized, 2.7–6.3°); holding error at neutral up to 3.6° (inside
  ADR's ±0.08 rad = ±4.6° offset).
- index_pip auto-calibration under-measured its stroke (63–74 vs 96 motor degrees), fixed by manual calibration
  → candidate: per-joint scale DR.
- The transformer student stalls the control loop for 216–593 ms in every trial (MLPs: one 209 ms stall); guess:
  TorchScript's profiling executor re-optimizing on the first real calls after a single warm-up call (unverified).

Consequence for the next round: model the servo (rate and acceleration limits, ~70–150 ms lag, lower bandwidth) in
the sim (actuator model / net, [actuator_net](actuator_net.md)), and make the action pipeline trackable (rate-limited
targets, lower EMA α or a stronger action-rate penalty); then re-fine-tune the deployable policies.

## Real-hand servo model as DR (2026-10-31)

Fit of the deploy session (W&B dataset `hand_trials_2026-10-31:v0`, `actuator/actuator_model_orca_v2_right_2026-10-31.json`;
chirps ±10° 0.1–5 Hz + 190 Hz steps of 2 / 5 / 15°): commanded angle → pure delay d (0–40 ms; + 0–10 ms state age
with the node) → **trapezoidal profile** with the Feetech profile register's acceleration (22.78 rad/s² motor side /
the joint's motor ratio = 14.8 rad/s² wrist, 23–30 fingers, 38–41 thumb_abd / thumb_dip; the speed register never
binds; a step of s peaks at √(a·s): 15° → ~150°/s) → stiff loop (fingers ωn 140–300 rad/s, wrist 20) with friction
**play** (hysteresis 2.4–3.8°, thumb_abd 5.8°) and a **slope** below 1 (index_pip 0.70, thumb_dip 0.80, others
0.87–0.95). Chirp RMSE 1.4–2.4° vs 5.6–6.1° raw. The sim's PD is heavily overdamped (ζ 3–160 at the joints' tiny
inertias, joint-space inertia at the neutral pose 1.8e-6–1.5e-3 kg m²): a first-order lag of Kd/Kp = 70–140 ms with no
dead time and no rate limit; a 1 rad error starts at ~10 rad/s.

Implemented (`adr.ServoProfileAction`, `adr.use_orca_v2_servo_model`): delayed EMA target → `trapezoid_step` (per
joint amax) → slope around the episode's start pose → play (backlash) → PD with damping = Kp × lag (fingers 10 ms,
wrist 100 ms). CPU test of the profile: step peaks within 2 % of √(a·s), no overshoot, arrival ~10 % earlier than the
continuous formula. Task `-asym-onedir-droppen-upright-servo-v0`: delay 2–14 substeps (17–117 ms), amax ×0.7–1.3, play
×0.5–1.5, slope ×0.9–1.1, encoder offset ±0.08 rad; policy interface unchanged. Eval `eval_policy.py --servo` (same
DR, outputs `*_servo`).

User instruction (12:15Z, via the deploy session): stop all pod trainings, use the whole GPU to fine-tune the best
speed / robustness mix under the servo model; candidates 07, 02, 02b, 01, 05 scored under `--servo` first (score =
RPM × drop-free), then the winner (07 if within 10 %) fine-tuned with 32k envs, servo eval every ~150 iterations.
Real-hand ratings so far: 07_tilt (no ADR) the only full 60 s run ("best till now"), 01 "bad", the others stopped
early after occasional drops.

**First servo-model scores (12:28Z, pod, `--servo`, 64 envs × 60 s, randomized as in training):**

| policy | RPM | drop-free | drops / trial | score RPM × DFSR | on the real hand |
|---|---|---|---|---|---|
| 07 tilt, no ADR (model_4000) | 5.38 | 3.1 % | 10.9 | 0.17 | only full 60 s run, rated best |
| 02 tilt + ADR (6998) | 6.77 | 6.2 % | 5.8 | **0.42** | stopped early |
| 02b tilt + ADR (5500) | 5.72 | 4.7 % | 5.8 | 0.27 | not run |
| 01 tilt + position ×3 + ADR (5500) | 6.10 | 4.7 % | 7.8 | 0.29 | rated "bad" |
| 05 tilt + position ×3 (3999) | 5.54 | 1.6 % | 8.0 | 0.09 | not run |

Every policy collapses (drops every 6–10 s), and the ranking contradicts the hand (07 survived 60 s on the hand,
here it is second to last): the model as implemented is too harsh or one part is wrong, i.e. **not validated yet**.
A bug (the helper `_scale` shadowed by `JointAction._scale`, reset crashed) was found by the pod and fixed (now
`_dr_scale`). Diagnosis: `eval_policy.py --servo_parts` switches delay / profile / slope / play / stiff loop on one at
a time; with none on it must reproduce the normal eval (sanity check of the class). Running on 07 and 02. Fine-tunes
under the full model meanwhile: 02 on the pod (32k envs, `02_tilt_adr_servo_ft_r6k`, since 12:28Z), 07 on DTU
(L40S, 29860426, pending).

**Trial replay validates the model without the slope (12:45Z).** `scripts/sim2sim/replay_hand_trial.py` feeds the
commanded joint targets of real-hand trials (node CSV logs, W&B `hand_trials_2026-10-31`) through each actuator model
and compares with the measured joints (kinematic: no contact load). Pooled over p1, p2, p3, p4, p7, RMSE [deg]:

| model | mean of 17 joints | wrist | index_mcp | thumb_mcp | middle_pip | ring_pip |
|---|---|---|---|---|---|---|
| raw command (q = target) | 15.8–23.1 per trial | | | | | |
| task's PD (first-order lag Kd/Kp, no delay) | 12.4 | 10.4 | 23.0 | 24.7 | 14.6 | 8.5 |
| delay + profile + stiff loop | 7.7 | 5.1 | 12.6 | 13.6 | 8.3 | 5.3 |
| **+ play (the new default)** | **7.2** | **4.7** | **12.0** | **13.0** | **7.8** | **5.0** |
| + slope (as first implemented) | 10.2 | 5.1 | 12.1 | 13.0 | 15.3 | 18.8 |

The acceleration-limited profile is real (the big slewing joints halve their error); the slope, scaled from the start
pose, is not (the measured slope is a small-amplitude friction effect; the deploy session's replay agrees). Default now
`adr.DEFAULT_SERVO_PARTS` = delay, profile, play, stiff. Residual: in trials the mcp / pip joints sit 8–12° more flexed
than any model predicts (cube load, not in the unloaded chirps). Hand ranking so far (deploy session): 07 (6.7°/s mean
wrist speed) full 60 s "best"; 02b (2.3°/s) full 60 s "ok, no drop"; 12 (74°/s) full 60 s "too much wrist"; 05
(51°/s) stopped at 46 s "bad"; 01, 03 "bad": wrist motion is the best predictor. Next: fine-tunes from 07 and 02b under
the corrected model + wrist velocity penalty (`-upright-servo-wristpen`, pod), re-scoring of all candidates under it.

**Independent confirmation (deploy session, 13:00Z).** After fixing its replay (a start-offset bias), the deploy
session's replay of p7, p02b, p4, p05, p12 agrees: delay + per-joint amax + stiff loop gives 4.1–7.1° RMSE per trial
(mean of 17 joints) vs 9.8–13.6° for the task's PD and 18.9–23.5° for the raw command, also on the joints whose delay /
loop are copied from the index finger (pinky_pip 1.8 vs 4.5, middle_mcp 6.8 vs 16.9, pinky_mcp 8.5 vs 22.3; wrist 2.0
vs 9.2, thumb_mcp 4.7 vs 20.4). Two more hand trials confirm the wrist finding: 18 (transformer student) "bad", wrist
50.7°/s; 16 (droppen, bidirectional) "bad, cube falls when the wrist moves", 33.6°/s. Hand ranking by mean wrist speed:
02b 2.3°/s (ok, no drop), 07 6.7 (best), 16 33.6, 05 50.6, 18 50.7 (bad), 12 73.8 (quite good, too much wrist).

**Per-part ablation of the servo model on 07 (model_4000; pod, 64 envs × 60 s, with the randomization ranges of
training):** all off 10.97 RPM / 87.5 % / 0.12 drops (sanity: the normal eval gives 11.21 / 93.8, within noise) · delay
17–117 ms only 12.66 / 45.3 % / 0.92 · **acceleration profile only 7.91 / 10.9 % / 3.64** · slope only (dropped since)
5.15 / 65.6 % / 0.50 · play only 10.85 / 95.3 % / 0.08 · stiff loop only (no profile) 3.79 / 34.4 % / 1.36. The
acceleration limit is what breaks the trained gait (it needs faster joints than the real servos deliver; matches the
real hand's 150–300 ms tracking lag and occasional drops); play is harmless; the stiff loop alone is not a realistic
setting (on the servo the profile is what smooths the motion). Fine-tunes under delay + profile + play + stiff loop +
wrist penalty, from 07 and 02b, 12288 envs each (32768 overflowed PhysX's contact buffer), started ~12:40Z.

**Closed loop says the model is too harsh (12:35Z).** Under delay 17–117 ms + profile + play + stiff loop (`--servo`
of f2e4e2a) every candidate drops the cube all the time (0 % drop-free, 07: 20 drops per minute) and 07, the hand's best,
ranks last. Two errors: (1) the delay was double counted: the measured pure delay is 0–40 ms (+0–10 ms ROS), the rest
of the 60–90 ms dead time is the profile's ramp, which the model now produces itself; (2) the stiff loop (damping cut
10×) removes the contact damping (alone it cost 07 / 02 60–70 points). Replay: profile + delay + play in front of the
task's own PD still fits at 8.2° (stiff 7.2°, task PD alone 12.4°). New default (2c18c33): delay + profile + play,
the task's PD; training delay 0–8 substeps (0–67 ms), amax ×0.8–1.2, play ×0.5–1.5; eval `--servo` = the measured
hand: delay 0–6 substeps, nominal values. Validation (07 / 02b above 01 / 05) and the two wrist-penalty fine-tunes
started together on the pod.

**Validation still fails with the corrected model (12:40Z, `--servo` of 2c18c33, the measured hand):** 01 11.24 RPM /
35.9 % / 1.58 (score 4.04) > 02 11.11 / 31.2 / 1.61 (3.47) > 02b 8.98 / 17.2 / 3.20 (1.54) > 05 8.03 / 15.6 / 4.14 (1.25)
> 07 7.52 / 6.2 / 5.66 (0.47). Much milder than before, but the ADR policies win and 07 is last; the hand says 07, 02b
good, 01, 05 bad. The model now rewards delay robustness, not what fails on the hand.

**The hand trials point at the palm angle (and so at friction).** Wrist command in the policy frame (p5 / p50 / p95):
the runs that lasted (p7 27.7 / 35.0 / 35.0, p2 14.0 / 35.0 / 35.0, p4 35 throughout) hold the wrist at its +35° limit,
i.e. keep the palm level (35° against the 30° mounting tilt); the bad ones (p1 −18.5 / 21.2 / 31.1, p3 −19.2 / 18.5 /
29.6) swing it and tip the palm down by up to ~50°. (The relay's wrist clamp is not involved: < 0.2 % of commands below
−35°.) Tipping the palm needs friction: the sim's finger–cube friction is 0.55–1.5; printed parts on the real hand are
likely lower. Test running: `--servo` with contact friction ~0.3–0.55 (hydra overrides of the two material events);
the deploy session is asked to measure the slip angle (μ ≈ tan θ). The wrist-penalty fine-tunes from 07 / 02b keep
running meanwhile (12:43Z, 12288 envs, 5.3 s/it).

**17 of 19 policies on the real hand (12:45Z, deploy session; flat forearm, −30° wrist relay): mean |wrist speed|
ranks them without exception.** 02b 2.3°/s full 60 s "ok, no drop" · 07 6.7 full 60 s "best" · 17 (asym ent 5e-3,
bidirectional) 15.2, 52 s "is ok" · 16 droppen 33.6 full 60 s "bad, cube falls when wrist moves" · 09 (tumbles) 47.0,
25 s · 05 50.6, 46 s "bad" · 18 student_pt 50.7, 44 s "bad" · 06 pos45 55.9, 23 s · 15 rotlin_droppen 57.2 full 60 s
"not very good" · 08 pos45 ent5e-3 61.9, 44 s · 13 rotlin 67.4, 24 / 34 s · 14 rotlin ent5e-3 72.1, 29 s · 12
rotlin_cont10k 73.8 full 60 s "quite good but too much wrist". Earlier: 01 "bad", 03 28 s, 04 58 s (wrist pinned like
02 / 07). The sim's own wrist speed (`cube_motion.py`, nominal rollout) orders them the same way (02 / 07 0.08–0.09
rad/s; 01 0.32; 03 0.52; tumbling / rotlin policies higher), so **sim wrist speed is a selection criterion validated on
the hand**, unlike the servo-model score so far. Ship rule for the fine-tunes: beat the base under `--servo` and keep the
sim wrist speed ≤ 07's (~0.1 rad/s).

**Cube motion under `--servo` (2c18c33, nominal servo):** 07 tilt 16.4°, 45 flips/min, **wrist 0.44 rad / 0.32 rad/s**
(normal eval 0.28 / 0.09; on the hand calm and pinned at its limit, 6.7°/s) · 02 10.4°, 13, 0.20 / 0.11 · 02b 11.9°, 24,
0.40 / 0.21 · 01 13.6°, 15, 0.30 / 0.30 · 05 16.8°, 36, 0.52 / 0.44. In closed loop the servo model makes 07's wrist
swing, which the real hand did not: the model still misses something that matters with the cube in hand (the mcp / pip
sit 5–13° more flexed under load on the hand than any model predicts). The validated wrist criterion is the normal
eval's (it ordered the 17 hand policies); ship rule: normal eval wrist speed ≤ ~0.1 rad/s with drop-free like the
base, and a better `--servo` score than the base. Note: `eval_policy.py` hydra overrides must come before
`--checkpoints` (nargs="+" takes everything after it as checkpoint paths).

**Real friction measured (12:50Z, deploy session):** cube on the open palm, hand tilted at the wrist until the cube
slides, 5 tries: 23 / 25 / 18 / 25 / 21° → **μ = tan θ ≈ 0.41 (0.32–0.47)**, below the sim's whole effective
cube–hand range (0.55–1.5 = PhysX average of hand 0.8–2.0 and cube 0.3–1.0). So a palm tipped more than ~20–25° loses
the cube on the real hand, which matches every trial (the level-palm policies survive). Finger pads and materials not
measured yet. Fine-tunes restarted with hand 0.3–0.7, cube 0.3–0.6 (contact 0.3–0.65; hydra overrides of the two
material events) + servo model + wrist penalty, from 07 and 02b (pod) and from 07 without the wrist penalty (DTU
29860492); re-score of the candidates at low friction with and without the servo model running (does friction alone
reproduce the hand ranking?).

**`--servo` at low friction (hand 0.3–0.6, cube 0.3–0.5; 13:50Z):** 02 3.46 (11.08 RPM / 31.2 %; wrist 0.21 rad /
0.12 rad/s) > 01 2.26 (9.03 / 25.0; 0.38 / 0.34) > 02b 1.61 (8.58 / 18.8; 0.36 / 0.16) > 05 1.23 (6.58 / 18.8; 0.72 /
0.68) > 07 0.32 (6.81 / 4.7; 0.66 / 0.46). Low friction costs 01 (4.04 → 2.26) as predicted, but 07 stays last. With
the wrist criterion added (sim wrist ≤ ~0.15 rad/s) the order is 02 > 02b, 01 / 05 / 07 out: matches the hand except
07, which in the sim is fragile under the actuator dynamics but ran a clean minute on the real hand (one trial).
Fine-tunes restarted 12:56Z with low friction + servo model + wrist penalty from 07 and 02b (`*_servo_wp_lf_ft_r6k`).

**Materials (user, 12:58Z): PLA cube, TPU palm, soft silicone fingers ("10 NV", probably Shore ~10A).** The measured
μ ≈ 0.41 is PLA on the TPU palm; PLA on soft silicone is typically ~0.8–1.2. One friction range for the whole hand is
therefore wrong: `use_real_hand_friction` sets the finger links to 1.2–2.0, the palm links (forearm, tower, carpals,
thumb_tp) to 0.3–0.6 via an extra startup event that overrides only their shapes, and the cube to 0.3–0.5, so PhysX's
average gives palm contacts 0.3–0.55 and finger contacts 0.75–1.25. Task `-upright-servo-wristpen-realfric-v0`
(c5a9993); the pod's fine-tunes from 07 and 02b switched to it (`*_servo_wp_rf_ft_r6k`). `--servo` at mid friction
(hand 0.5–0.8, cube 0.4–0.6): 02 3.79 > 01 2.90 > 02b 2.49 > 05 1.13 > 07 0.41: under the servo model 07 is last at
every friction level.

**Finger pads measured too (13:00Z):** PLA cube on the silicone pads slips at 30–40° → μ 0.58–0.84. Friction ranges
set to the measurements (2b253ad): finger links 0.8–1.3 (contacts 0.55–0.9), palm links 0.3–0.5 (contacts 0.3–0.5), cube
0.3–0.5. The pod restarts the two fine-tunes on it (`*_servo_wp_rf_ft_r6k_v2`).

**Friction alone reproduces the hand (13:10Z, normal eval at low friction, hand 0.3–0.6 / cube 0.3–0.5):** 07 9.74
(11.13 RPM / 87.5 % / 0.20 drops; wrist 0.16 rad / **0.14 rad/s**) > 05 8.71 (11.15 / 78.1; **0.46**) > 01 8.67 (11.56
/ 75.0; **0.47**) > 02b 6.42 (8.22 / 78.1; 0.11). 07 first, and the wrist speed separates the hand's good (07, 02b) from
its bad (01, 05) policies exactly. The servo profile in closed loop is what ranked 07 last, so it is not validated
(the kinematic replay is). Hedge: the 07 fine-tune switched to `-upright-wristpen-realfric-v0` (5de6236: real-hand
friction + wrist penalty + plain 0–100 ms delay + encoder offset, no profile); the 02b fine-tune stays on
`-upright-servo-wristpen-realfric-v0`.

**Skeleton material (user):** the printed skeleton is PLA, "extreme low friction" on the PLA cube (not measured;
~0.1–0.25 suggested). The hand's 53 colliders are all under instanced prims (no USD materials, no per-prim binding),
so per-collider friction needs a PhysX shape-index mapping (+ GPU check); per link, the skinless links (the AP links,
thumb_tp / thumb_ap) can be made PLA. Contact friction below half the cube's needs PhysX's "min" combine mode instead of
the default average. Next round.

**Repeat runs on the hand (13:20Z):** the user re-ran the top 4 and ranked them 07 > 02b > 17 > 12. 07: full 60 s in
both runs; 02b: run 1 "ok, no drop", run 2 "quite bad" (21 s), wrist calm in both: inconsistent; 17 40 s; 12 36 s. 07
is the base (it also tops the low-friction Isaac ranking). The 02b servo-profile fine-tune is replaced by a second seed
of the 07 friction-only fine-tune (`07_wp_rf_ft_seed1_r6k`). Ship rule: beat 07's low-friction eval (9.74 score, 87.5 %
drop-free) with the wrist at ≤ ~0.1–0.14 rad/s.

**Hand findings report (deploy session, 13:25Z):** on the user's Mac, `~/Desktop/Studium/Master/RWR/2026-10-31_ORCA_v2_hand_findings/`
(FINDINGS.md, trials_overview.csv with 27 trials, actuator PDF + JSON, measurement log); raw data W&B
`hand_trials_2026-10-31` (v2; v3 adds the repeat trials). Beyond the items above: start lag 51–88 ms on the hand vs
2–3 ms in sim with similar 90 % times (160–330 ms); ROS adds 0–7 ms; holding error ≤ 3.6°; noise ~0; **pinky motors
reach 68–74 °C after ~35 min holding a pose** (keep torque off when idle). Open: the 07 fine-tune on the hand,
per-collider friction (skin vs PLA skeleton), why the closed-loop servo profile mis-ranks, a higher servo acceleration
register (untested), the index_mcp / middle collision angle.

**Next-round friction model (prepared; task `-upright-wristpen-realfric2-v0`):** the skinless finger links (the
four AP links, thumb_tp, thumb_ap: all hulls PLA) become a third class. With PhysX's average combine and the cube at
0.2–0.3, link ranges 2 × contact − cube give contacts: TPU palm (forearm, tower, carpals) 0.3–0.5, silicone-skinned
links 0.55–0.9, PLA links 0.1–0.25 (CPU check: exact ranges, events ordered after the hand-wide one). The skinned
links' own PLA hulls still get the skin value (instanced colliders). Not started: the current fine-tunes run on
`-realfric`.

**First real-hand fine-tune shipped (13:30Z): W&B `orca_v2_zrot_hand_ft:v0` (alias `07ft_s0_m4150`).** 07 model_4000
fine-tuned 150 iterations (12288 envs, pod) on `-upright-wristpen-realfric-v0`. Normal eval at the measured low
friction (`-upright-v0` + hand 0.3–0.6 / cube 0.3–0.5): 10.04 RPM, 92.2 % drop-free, 0.08 drops, wrist 0.16 rad / 0.08
rad/s, tilt 5.9°, 2.2 flips/min vs the 07 base 11.13 / 87.5 / 0.20 / wrist 0.14: fewer drops and a calmer wrist at −10 %
speed. On its own task (delay / offset DR, per-link friction): 13.07 RPM, 84.4 %, wrist 0.08. Export bit-exact; the
unchanged course node ran it 10 s in the MuJoCo loop (no drop, interface match ≤ 6e-7). Sent to the deploy session.

**Seed 1 reproduces it (13:35Z):** `07_wp_rf_ft_seed1_r6k` model_4150 at the measured low friction: 10.34 RPM, 93.8 %
drop-free, 0.06 drops, wrist 0.08 rad/s (score 9.70 vs 07's 9.74); own task 13.17 RPM, 81.2 %. Both seeds cut the drops
~3× and calm the wrist at about the base's speed. Shipped together as `orca_v2_zrot_hand_ft:v2` (folders
`07ft_s1_m4150`, recommended, and `07ft_s0_m4150`; v1 had a misnamed folder, superseded).

**Does the sim at the measured friction fit the hand? (8 policies, 13:50Z)** Normal eval at hand 0.3–0.6 / cube 0.3–0.5
(pod, `--tag _lowfric`, cube_motion) vs the hand trials (mean |wrist speed| from the node CSVs; duration of the trial):

| policy | sim wrist rad/s | hand wrist rad/s | sim drop-free % | hand duration s (mean) |
|---|---|---|---|---|
| 02b | 0.11 | 0.06 | 84.4 | 37.4 (56.8 / 18.0) |
| 07 | 0.14 | 0.09 | 87.5 | 55.7 (56.7 / 54.7) |
| 02 | 0.16 | 0.22 | 81.2 | 44.3 |
| 05 | 0.46 | 0.88 | 78.1 | 43.3 |
| 01 | 0.47 | 1.15 | 75.0 | 16.4 |
| 06 | 0.49 | 0.98 | 70.3 | 20.3 |
| 03 | 0.73 | 0.89 | 87.5 | 27.5 |
| 08 | 0.79 | 1.08 | 79.7 | 40.9 |

Spearman: sim vs hand wrist speed **ρ = 0.83 (p = 0.01)**; sim drop-free vs hand duration 0.51 (p = 0.19); sim wrist vs
hand duration −0.45 (p = 0.26). The sim reproduces *which* policies move the wrist (the swingers about 2× more on the
hand: slipping makes them correct harder), not yet *how long* they last (single, user-stopped trials are noisy). The
other 11 policies are evaluated on DTU (job 29860553) for n = 19.

**All 19 policies (16:20Z, DTU eval11 added: 04, 09–18).** Same setting; hand wrist = mean |wrist speed| over the
trials (11: the three trials with the offset), hand duration = mean logged seconds (user-stopped, capped at 60 s):

| policy | sim wrist rad/s | hand wrist rad/s | sim drop-free % | hand duration s | sim RPM |
|---|---|---|---|---|---|
| 16_droppen | 0.06 | 0.59 | 92.2 | 57.3 | 8.57 |
| 17_asym_ent5e-3 | 0.08 | 0.39 | 79.7 | 42.9 | 7.59 |
| 18_student_pt | 0.08 | 0.88 | 89.1 | 40.0 | 7.24 |
| 04_tilt_wristpen | 0.11 | 0.00 | 70.3 | 53.9 | 5.99 |
| 11_student_pt_rotlin | 0.12 | 0.79 | 85.9 | 29.9 | 11.67 |
| 15_rotlin_droppen_ent5e-3 | 0.12 | 1.00 | 81.2 | 57.0 | 9.89 |
| 12_rotlin_cont10k | 0.44 | 1.28 | 76.6 | 45.7 | 11.40 |
| 14_rotlin_ent5e-3 | 0.62 | 1.26 | 85.9 | 25.6 | 11.04 |
| 13_rotlin | 0.72 | 1.16 | 79.7 | 25.9 | 11.01 |
| 09_onedir_droppen_tumbles | 2.13 | 0.82 | 42.2 | 22.4 | 15.32 |
| 10_onedir_clip3_tumbles | 2.25 | 1.21 | 26.6 | 25.9 | 14.98 |

(plus the 8 above). Spearman, n = 19: sim wrist vs hand wrist **ρ = 0.61 (p = 0.006)**; **sim wrist vs hand duration
ρ = −0.63 (p = 0.004)**; sim drop-free vs hand duration 0.40 (p = 0.09); hand wrist vs hand duration −0.43 (p = 0.07).
Without 04 (its hand wrist barely moved, 0.1 °/s): sim vs hand wrist 0.57 (p = 0.01).

Reading: (1) the sim wrist speed at the measured friction predicts how long a policy lasts on the hand better than any
other sim number, and better than the hand's own wrist speed. (2) The match is family-dependent: every tilt-penalty
(`-upright`) policy (02, 02b, 04, 07) is calm in sim *and* on the hand, while the five older calm-in-sim policies
without the tilt penalty (16, 17, 18, 11, 15: sim 0.06–0.12 rad/s) swing 0.4–1.0 rad/s on the hand. Without the
penalty, a policy that is calm in sim can still swing the wrist on the real hand, so a calm sim wrist only counts as
evidence together with the tilt penalty. (3) The two sim tumblers (09, 10) are the only ones the sim rates as worse
than the hand does. The ship rule (measured-friction eval, wrist ≤ ~0.1 rad/s, tilt-penalty recipe) stands.

**Shipped `orca_v2_zrot_hand_ft:v4` (13:50Z):** `07ft_s1_m4300` (recommended): at the measured friction 10.80 RPM, 93.8 %
drop-free, 0.06 drops, wrist 0.08 rad/s, score 10.12 vs 07's 9.74: beats the base on every criterion; own task 13.64
RPM / 84.4 %. Plus `07ft_s1_m4150`, `07ft_s0_m4150`.

**New best real-hand fine-tune (14:15Z): `07ft_s0_m4600`, W&B `orca_v2_zrot_hand_ft:v6` (latest; v5 holds only this
checkpoint).** Seed 0 model_4600 at the measured friction: 11.29 RPM, 96.9 % drop-free, 0.05 drops, wrist 0.16 rad / 0.08
rad/s, tilt 5.7°, 1.5 flips/min (score 10.93 vs 07's 9.74: as fast, 4× fewer drops, calmer wrist); own task 13.99 RPM /
90.6 %. Other checkpoints so far at the measured friction: s0 m4150 10.04 / 92.2, m4300 10.72 / 90.6, m4450 11.40 / 85.9;
s1 m4150 10.34 / 93.8, m4300 10.80 / 93.8, m4450 10.89 / 89.1; realfric2 (PLA skeleton) m4200 10.30 / 81.2 (own task
13.09 / 89.1).

**Fine-tunes plateau; candidate B added (15:25Z).** At the measured friction: seed 0 m4900 11.39 / 96.9 / 0.03, m5050
11.67 / 93.8, m5200 11.17 / **98.4** / 0.02 (own task 12.11 vs 12.68 for m4600); seed 1 m4900 11.36 / 95.3, m5050 11.38 /
93.8, m5200 10.90 / 93.8; realfric2 (PLA skeleton) m4600 11.39 / 87.5, m4800 11.19 / 92.2, m5000 11.45 / 84.4, m5200 11.07 /
**98.4** / 0.02. Seed 0 m4600 stays recommended (A: best on its harder training task, 90.6 %); realfric2 m5200 is
candidate B (trained also with the slippery PLA skeleton). `orca_v2_zrot_hand_ft:v8` (latest) holds both plus the
earlier three; v7 holds only B.

## Cube rotation-axis region (user, 2026-10-31 15:30Z)

The user chose, with `docs/ablation/tools/palm_axis_region.html`, the region the cube's rotation axis (the vertical
through the cube centre, the cube turns about world z) may occupy: centre (−11.6, 13.6) mm, radius 27.5 mm (env frame;
contains the commanded cube position (0, 0) at 17.9 mm). Term (`rewards_ablation.cube_outside_region`, user: "fall
exponentially if outside the circle"): 50 × (exp(−e / 10 mm) − 1), e = distance outside the circle: 0 inside, −31.6 per
second 10 mm outside, slope 5 per mm at the boundary, saturating at −50 per second (rotation reward ~90 per second).
Task `-upright-wristpen-realfric-region-v0` (78e09b8) = the real-hand recipe + this term. Baseline: 07 (model_4999)
keeps the cube inside 91.8 % of the time (`cube_motion.py --region -0.0116 0.0136 0.0275`). Fine-tune from
`07ft_s0_m4600` on the pod (`07ft_region_ft_r6k`, 1000 it), target > 98 % inside at the same speed / drop-free.

Region fine-tune `07ft_region_ft_r6k` started 15:33Z on the pod (resume 07ft_s0_m4600, 8192 envs, ~7.6 s/it; config
check: cube_region weight 50, σ 10 mm, centre / radius as chosen). Baseline of its base checkpoint (LOWFRIC rollout):
94.1 % of alive steps inside the circle.

m4700 (100 it into the term): LOWFRIC 11.77 RPM / 93.8 % drop-free / 0.08 drops, 93.2 % in region; OWN 14.28 / 82.8 / 0.22, 90.4 % in region: no region gain yet. DTU queued two more region fine-tunes: from candidate B (rf2 m5200) and from A with seed 1. Final 07 fine-tune checkpoints (m5400/m5499, all three seeds/variants) decline to 84–91 % drop-free at the measured friction; the shipped m4600 / rf2 m5200 stay best.

m4800 / m4900: LOWFRIC 11.55 / 95.3 % / 0.05, 92.9 % in region; 11.58 / 92.2 % / 0.09, 94.2 % (OWN 91.0 %). The term is active but small: Episode_Reward/cube_region −1.8 per episode vs rotation_progress +67 (~3 %, like the tilt term), and the actor sees no cube pose, so it gets a weak, indirect signal. Next: `-realfric-region200-v0` (weight 200, −126 per second at 10 mm outside, −200 saturated, above the ~90 per second of rotation reward). The weight-50 run stops at m5000 as the control; region200 from 07ft_s0_m4600 on the pod; on DTU from B (rf2 m5200) and from A with seed 1.

Control (weight 50) final m5000: LOWFRIC 11.41 / 93.8 % / 0.06, 93.7 % in region (OWN 13.97 / 87.5 %, 91.0 %): 400 iterations did not move the in-region rate. `07ft_region200_ft_r6k` started 16:18Z on the pod (config: weight 200, r 27.5 mm, σ 10 mm; ~4.8 s/it).

region200 m4700 (100 it): LOWFRIC 11.46 RPM / 96.9 % / 0.03 drops, wrist 0.07, **95.0 % in region** (base 94.1 %, weight 50 at the same iteration 93.2 %); OWN 13.65 / 87.5 %, 92.9 % in region. First movement, at no cost in speed or drops.

m4800: LOWFRIC 11.75 / 92.2 % / 0.09, 95.3 % in region (OWN 94.1 %). **Where the cube leaves the circle** (rollouts of 07ft_s0_m4600 and 07rf2_s0_m5200 at the measured friction, 64 envs × 60 s): 95–97 % of the out-of-region steps leave towards −y (bearing −90° to −45° from the centre); median depth 3 mm, p90 18 mm; only 0.2–1 % fall within 1 s before a drop; every env leaves at times. The cube sits on average at (−6.9, −6.0) mm, 20 mm from the centre (−11.6, 13.6) mm: only 7.5 mm margin on that side. The position-tracking reward still pulls to the command target (0, 0, −0.05), which lies 17.9 mm from the centre in exactly that direction, so the tracking term and the region term pull against each other; a larger region weight only fights that. **`-realfric-region200c-v0`** (311254f): region200 with the tracking reward's x-y target moved to the region centre (`rewards_ablation.track_position_exp_xy`). The command and the policy's goal observation stay [0, 0, −0.05, 0, 0, ω], which the deploy node hard-codes. On the pod from 07ft_s0_m4600 (control: the region200 run, same base and seed); on DTU with seed 1 (job 29861178).

region200c m4700 (first 100 it, 16:47Z start as 5th process): LOWFRIC 10.95 / 84.4 % / 0.27, **95.7 % in region** (region200 needed 300 it for that); OWN 13.70 / 82.8 %, 93.6 %. Drop-free dipped (the gait moves the cube 18 mm to the new target); watch m4800–m5000. region200 m5000: 11.56 / 95.3 % / 0.05, 95.5 % (OWN 94.6 %): the plain weight plateaus at ~95.5 %.

**512-env re-eval of the candidates (pod, 17:23–17:40Z, `lowfric512_summary.md` on remote-results):**

| candidate | RPM | drop-free % | drops/min | wrist rad/s | in region % |
|---|---|---|---|---|---|
| A 07ft_s0 m4600 | 11.41 | 90.0 | 0.109 | 0.06 | 93.2 |
| B 07rf2_s0 m5200 | 11.15 | 89.3 | 0.129 | 0.07 | 91.5 |
| 07ft_s0 m5200 | 11.16 | 91.4 | 0.100 | 0.06 | 89.7 |
| 07ft_s1 m4900 | 11.39 | 92.2 | 0.082 | 0.07 | 94.2 |
| region200c m4800 | 11.48 | 90.0 | 0.111 | 0.07 | 94.7 |
| region200 m5000 | 11.40 | 90.6 | 0.105 | 0.07 | 95.2 |
| roll10 m4600 | 11.48 | 91.6 | 0.102 | 0.07 | 93.3 |
| roll10 m5000 | 11.80 | 90.0 | 0.113 | 0.08 | 90.6 |

All within ~2 SE of each other. The 64-env numbers we selected on (A 96.9 %, B 98.4 %, region 96–97 %) were winners of many noisy draws. The region terms gain only 1.5–2 points in region (94.7 / 95.2 vs 93.2 %), far from 98 %. Next: `-region200cs` (6b5f4b9) also spawns the cube at the region centre (from (0, 0) it starts 7.5 mm inside the circle on the side where it leaves), queued first on the pod. The eval also runs with a centred spawn, because the real cube is placed by hand. Ship rule now uses the 512-env eval ([deployable](deployable.md#real-hand-fine-tunes-2026-10-31-afternoon)).

**Where the drops come from (pod, 2026-11-01 01:30Z, 512-env LOWFRIC rollouts of A m4600 and kl005 m4900, `drop_analysis.md` on remote-results):** (1) most of the drop-free loss is the catch right after the eval's spawn: of the envs that drop, 34/51 (A) and 23/43 (kl005) drop within the first 2 s, nearly all at 0.25–0.75 s. Drop-free over t ≥ 2 s is 96.5 % (A) and 96.1 % (kl005), vs 90.0 / 91.6 % overall. On the hand the cube is placed by hand, so the steady-state number is the relevant one (`cube_motion.py --steady`). (2) Mid-episode drops follow tilt: in the second before a drop the tilt averages 10–12° (alive 5.9°), max 24–27°. Drop hazard per 1000 steps: tilt < 10° 0.25, 10–15° 0.8–1.0, 15–25° 5–11, ≥ 25° 60–68. Distance from the region centre is normal (19–20 mm), exit bearings are spread over all sides, and the −y edge raises the hazard only 2–3×. So the region work will barely cut drops; tilt control will. Queued: `07_wp_rf_kl005_tilt400_ft` (tilt −400 on the kl005 recipe), and `06_wp_rf_calm_ft` already tests 2× tilt on the fast policy.

Drop-free after 2 s for the first 512-env set (overall → t ≥ 2 s): A 90.0 → 96.5, B 89.3 → 94.7, 07ft m5200 91.4 → 95.5, 07ft_s1 m4900 92.2 → **98.0**, region200c m4800 90.0 → 96.9, region200 m5000 90.6 → 96.3, roll10 m4600 91.6 → 96.9, m5000 90.0 → 95.9. SE at ~97 % with 512 envs is ±0.75 %, so the best (s1 m4900) vs A is 1.4 SE: not yet a ship by the > 2 SE rule.

DTU 512-env eval (job 29862900, same setting; overall drop-free / RPM / in region / after 2 s): region200c seed 1 m5500 90.2 / 11.05 / 97.0 / 95.3, **m5599 90.2 / 10.95 / 97.2 / 95.5**; rf2 region200 m6199 88.9 / 11.08 / 96.6 / 95.3; 07_wp_rf seed 2 m5499 90.6 / 11.24 / 90.9 / 96.5; rf2 seed 1 m4500 89.8 / 11.33 / 91.5 / 95.5. The recentred region term reproduces across seeds: +4 points in region at 512 envs (97.2 vs A's 93.2 %) at the same overall drop-free, −0.5 RPM, −1 point after 2 s (n.s.). Still under the 98 % region target.

**06_calm (2026-11-01 02:15Z), the fast policy made calmer:** 06_wp_rf (pos45 base + real friction, 15 RPM, tilt 14°, wrist 0.23) fine-tuned with tilt −400 and wrist −8 (2× each). m5800 at 512 envs, measured friction: **13.52 RPM**, 88.3 % drop-free, **97.9 % after 2 s**, **98.8 % in the user's circle** (the first checkpoint over the 98 % target), tilt 8.6°, wrist 0.152 rad/s. After the spawn catch it is as reliable as the 07 line and 2 RPM faster. It still breaks the wrist rule (≤ ~0.1; the 07 line sits at 0.06–0.08), and its tilt is above the 07 line's 5.7–6.4°, and tilt is the mid-episode drop driver. Hand evidence for this wrist range is mixed: 07 had sim wrist 0.14 at the measured friction and was the best on the hand, while 02 (0.16) lasted 44 s. Not shipped yet: `06_calm2` (wrist −16, from 06_calm's final checkpoint) is queued to close the wrist gap.

06_calm2 (wrist −16 L2, from m5800) m6200 at 512 envs: 13.38 RPM, 87.3 %, 97.5 % after 2 s, 98.9 % in region, tilt 8.1°, wrist 0.148 (from 0.152). Doubling the wrist weight did nothing because the term is L2 (`joint_vel_l2`): at 0.15 rad/s, −16·q̇² costs < 1 per second against ~90 of rotation reward. Squared penalties only bite on large swings, which is why they fixed the 0.5–2 rad/s swingers but cannot take 0.15 down to 0.1. New task `-realfric-wristl1-v0` (0f03067) adds an L1 term (`joint_vel_l1`, −40: ~6 per second at 0.15 rad/s, a constant pull near zero). `06_calm3_l1` on the pod from calm2 m6200 (tilt −400); on DTU from 06_calm m5800 with tilt −800 (29863258), to isolate the tilt effect.

**Wrist speed over all envs (2026-11-01 03:55Z).** cube_motion's wrist speed came from the first 4 envs only (eval_policy logs link poses for --video_envs, default 4). With --video_envs 512: A m4600 0.079 rad/s (4-env figure 0.06), 07ft_s1 m4900 0.075. The ship rule's "wrist ≤ ~0.1" now means the all-env number: the 07 line sits at ~0.08. All earlier wrist numbers in these docs, including the 19-policy sim-vs-hand table, are 4-env figures (same bias for every policy, so the ranking holds). 512-env runs also move about ±2 points between repeats (seed 1 m4900: 92.2 / 98.0 % before, 90.4 / 96.9 % now, only --video_envs differs), so differences under ~2 points are noise. 06_calm3_l1 m6400 (the dip right after the L1 switch): 12.97 RPM, 84.8 %, 96.5 % after 2 s, 98.8 % in region, tilt 7.1°, wrist 0.138 (all envs); own task 15.84 RPM / 84.6 % / 96.3 % after 2 s.

06_calm3_l1 every checkpoint at 512 envs (all-env wrist; refs A 0.079, 07ft_s1 0.075, region policy 07region_s1_m5500 0.07 on DTU): m6500 12.58 RPM / 88.7 % / 98.8 % after 2 s / 99.0 % in region / tilt 6.7° / wrist 0.122; m6600 12.53 / 86.7 / 96.9 / 98.9 / 6.3° / 0.116; **m6700 12.83 / 89.1 / 97.5 / 99.0 / 6.6° / 0.114** (own task 15.95 RPM / 86.1 % / 97.5 %). Same drop-free as A within noise, +1.4 RPM, +6 points in region, tilt at the 07 line's level. Its wrist is 1.4× the baselines. But the original 07, which ran the full 60 s twice on the hand, read 0.14 rad/s on the 4-env measure, about 0.18 on the all-env one (A: 4-env 0.06 → all-env 0.079, ×1.3). So 0.11 is calmer than the best hand-tested policy. DTU evals are bit-repeatable (07region_s1_m5500 rerun: identical 11.05 / 90.2 %); the pod's move ~±2 points between repeats.

**Tilt penalty ×2 on the 07 line does not cut drops** (`07_wp_rf_kl005_tilt400_ft`, 512 envs, all-env wrist): m4600 11.52 RPM / 90.2 % / 96.7 % after 2 s / 92.2 % in region / tilt 6.1° / wrist 0.075; m5200 11.41 / 90.8 / 96.3 / 91.2 / 5.6° / 0.074, identical to A (90.0 / 96.5 / 6.1° / 0.079). The 07 line already holds the cube at ~6°, where the drop hazard is lowest, so its remaining drops come from elsewhere. **On the 06 line, tilt −800 is what calms it:** `06_calm3_l1_tilt800` m6200 12.66 RPM / 86.5 % / 98.0 % after 2 s / 98.9 % in region / tilt 5.0° / wrist 0.090, vs w80 (L1 −80, tilt −400) m7100 13.65 / 90.8 / 98.8 / 98.7, tilt 10.6°, wrist 0.105. A stronger wrist penalty alone shifts the motion into cube tilt; the tilt term stops that.

**Replication and the last calm-line comparisons (2026-11-01 06:30Z, 512 envs, all-env wrist).** 06_calm3_l1 seed 1 m7000: 12.97 RPM / 89.5 % / 99.0 % after 2 s / 99.0 % in region / tilt 8.3° / wrist 0.115, vs seed 42 (shipped as rank 2) 12.93 / 90.0 / 98.2 / 98.8 / 7.2° / 0.113: reproducible, seed 1 tilts 1–2° more. tilt800 seed 1 tracks seed 42 (m6000 12.26 / 86.7 / 5.9° / 0.110 vs 12.30 / 88.9 / 5.6° / 0.108; m6100 12.34 / 88.1 / 5.1° / 0.095 vs 12.25 / 85.4 / 5.2° / 0.099). Combined L1 −80 + tilt −800 (`06_calm3_w80_t800`) m6600–6800: 12.0–12.15 RPM, 88 %, tilt 5.1–5.9°, wrist 0.084–0.088. It is as calm as tilt800 m6300 but 0.3–0.5 RPM slower, so the extra L1 adds no calm beyond tilt −800. The final calm recipe is tilt −800 + L1 −40 (rank 1).

**The calm recipe works in one stage (06_oneshot, 2026-11-01 07:40Z).** tilt −800 + L1 wrist −40 + L2 wrist −16 + tracking 45 on `-realfric-wristl1-v0`, straight from the raw 06 base (pos45 m3999, tilt 18°, wrist 0.49): m4500 (500 it) at 512 envs 12.07 RPM / 91.8 % / 98.4 % after 2 s / 98.6 % in region / tilt 4.7° / wrist 0.081, own task 14.59 / 87.5 / 98.4. That is rank-1 level (12.46 / 89.5 / 98.4 / 99.0 / 4.8° / 0.086) after one stage instead of four. The recipe, not the staged path, does the work.
m4600: 12.49 / 89.6 / 98.0 / 98.7 / 4.5° / 0.080, own task 14.74 / 88.3 / 98.6: rank 1 on every number after 600 iterations. The recipe also bites on a second fast lineage (upright_pos45 seed 2, tilt 11.6° → 7.7°, region 69 → 92 % within 100 iterations, 64 envs).

**A higher command re-maps the speed, it does not raise it.** Rank 1 fine-tuned toward 2.0 rad/s: at m6400 it turns slower at 2.0 (8.6 RPM). After adapting (m6500–6600) it reaches 12.5 RPM at 2.0, rank 1's speed at 1.5, while its own speed at 1.5 falls to 11.4. The speed ceiling is the gait, not the reward clip.

**The calm recipe generalises across lineages.** One stage on upright_pos45 seed 2 (13.6 RPM, tilt 11.6°, wrist 0.31, 69 % in region): final m5498 at 512 envs 12.56 RPM / 91.4 % / 97.3 % after 2 s / 96.7 % in region / tilt 4.0° / wrist 0.078, close to rank 1 (13.07 / 89.5 / 98.8 / 98.9 / 4.1° / 0.074). The 06 base still gives the best region and after-2 s numbers. With 3× tracking the 07 base stays at A's speed and region (11.1–11.2 RPM, 94–95 %), so that advantage comes from the 06 base's grasp.

**Deploy branch (user's standing instruction, ~15:35Z):** every new policy that passes the ship rule also goes to
`rwr-fast/rwr_system`, branch `deploy-policies-2026-10-31`, folder `deploy_policies/` (policy.pt in Git LFS, SPEC.md,
manifest.json entry, README rank), pushed with `scripts/export/deploy_to_rwr_system.py spec.json --push` (dry run
without `--push`; verified: LFS pointer, re-ranking, commit identity). Already there: 07ft_s0_m4600 (rank 1),
07rf2_s0_m5200 (2), 07_tilt, 02b_tilt_adr_m5500, 17_asym_ent5e-3, 12_rotlin_cont10k.
