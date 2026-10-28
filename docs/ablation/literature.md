# Literature (cross-cutting)

[Index](README.md). Topic-specific literature lives in the topic files: reward structures in [rewards](rewards.md),
HORA code details in [RMA](rma.md), student fine-tuning in [teacher-student](teacher_student.md#student-ppo-fine-tuning-background-from-the-old-notes-15).

## ORCA Proprioceptive Transformer paper takeaways

arXiv 2605.21330, ETH SRL, IROS 2026. Their codebase is the base of ours: same Isaac Lab setup (8192 envs, 120 Hz /
20 Hz, 32 s episodes), same teacher MLP [512, 512, 256, 128], same reward weights (direction 60, magnitude 30, action
rate −0.075), same 81-dim teacher obs (= our `TeacherObsGroupCfg`), teacher 10k PPO iters at lr 1e-4.

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
   history. The repo's `DistillationRunner` / teacher–student leftovers are the intended path. → [teacher-student](teacher_student.md)
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
   per-step). Paper doesn't give σ; start with ±0.02–0.05 rad ([sim2real](sim2real.md#actuator-and-observation-models)).
8. **Confirmed choices we already have**: action-rate penalty ("critical" for smooth hardware motion), robot friction
   0.8–2.0, cube friction 0.3–1.0, cube mass 0.5–1.5. Their link mass scale is 0.95–1.05; check ours.
9. Wrist: fixed at 0 on hardware (no sensor); policy still outputs 17 joints → see fixed-wrist ablation ([sim2real](sim2real.md)).
10. Cube: they use 55 mm (also 65 mm); we have 45 mm → **confirm demo cube size**.
11. **Eval protocol to copy**: 3 × 60 s real trials; RPM, rotation accuracy, drop-free success rate, drop count
    ([eval protocol](eval_protocol.md)).

## In-hand rotation literature review

Subagent, 2026-10-27; summaries of paper text, [S] = snippet only.

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

Where they went: (1) [history](history_and_memory.md); (2), (9) [rewards](rewards.md); (3) deprioritized, see
[rewards](rewards.md) log 2026-10-28 02:00; (4), (5), (6) [sim2real](sim2real.md); (7) [RMA](rma.md);
(8), (10) [action space](action_space.md).

## Actuation, sensing, vision literature review

Subagent, 2026-10-27; PDFs read in full unless noted.

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
(3) servo velocity / slew limit on targets; (4) history encoder (see [teacher-student](teacher_student.md));
(5) motor-side obs model (q + τ/k_tendon, k randomized — untested idea), keep Faive's ×0.5 as a deploy knob;
(6) sweep EMA α at deploy; (7–9) Subtask 2: train with pose hold/delay/injection noise, run the estimator at a fixed
integer rate with outlier rejection instead of heavy smoothing, distil vision into a history student; (10) backlash
model from chirp reversals.

## Paper → run mapping and code availability

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

HORA details that matter for us (from the code) are in [RMA](rma.md#literature). Reward literature (2026-10-28):
[rewards](rewards.md#literature-reward-structures-for-in-hand-rotation-subagent-2026-10-28). Next-architecture
literature (SimBa 2410.09754 / BRO): [PPO and capacity](ppo_and_capacity.md).
