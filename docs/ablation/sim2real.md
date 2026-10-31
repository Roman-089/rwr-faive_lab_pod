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
