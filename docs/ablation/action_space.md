# Action space and control (EMA α, relative actions, control frequency)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

- EMA α 0.2 at 20 Hz is a ~5-step (≈250 ms) time constant on joint targets; it may cap gaiting speed.
- Relative joint targets make small, fast finger motions easy to express.
- Observation behind both: **every policy rotates faster under the perturbed eval** (A, B, C, LSTM: +0.3 to +1 RPM)
  while dropping 3–8× more; the 1–2-step delay apparently acts as extra low-pass / phase lag on finger gaiting
  ([asymmetric critic log, 2026-10-28 06:45](asymmetric_critic.md#log)). A's late checkpoints rotate *faster* with the
  test-time delay (0.6 → 1.2 RPM): hint that the late policy is jittery and a low-pass (delay) helps — worth checking
  action-rate / EMA alpha. Action smoothing is a control choice, not part of the Phase 1 architecture study.

## Current settings

| Parameter | Current | To explore |
|---|---|---|
| EMA action smoothing `alpha` | 0.2 | 0.3, 0.5 (faster response) |
| Control frequency | 20 Hz (decimation 6 @ 120 Hz) | keep unless the real loop runs faster — **fixed at 20 Hz by the course** (sim and real robot, task PDF) |
| Spin command range | 0.9–1.5 rad/s, resampled every 8 s | curriculum: start slow, increase |
| Episode length | 32 s | keep |

## Literature

- HORA / LEAP: relative joint targets (q_target = q_target,prev + s·a). Ranked idea (8) delta actions and (10)
  runtime-tunable action scale / grasp offset on deploy, from the [in-hand rotation review](literature.md#in-hand-rotation-literature-review).
- Visual Dexterity 2211.11744: EMA 0.8. DeXtreme: smoothing only if no added latency.
- Actuation review item (6): sweep EMA α at deploy ([literature](literature.md#actuation-sensing-vision-literature-review)).
- DLR Sievers 2022: action low-pass, filter 0.12→0.5 s curriculum ([rewards](rewards.md)).

## Runs (next architecture candidates, agreed 2026-10-28)

Base = the winner of the reward wave (B, `-rotlin` or `-rotsoft`, decided by eval). Single change each.

| Run | Change | Why | Status |
|---|---|---|---|
| `alpha05` | EMA α 0.2 → 0.5 | α 0.2 at 20 Hz is a ~5-step (≈250 ms) time constant on joint targets; may cap gaiting speed | queued on rotlin + ent 5e-3 (`abl_B_rotlin_ent5e-3_alpha05`, 29845928 / L40S) |
| `relact` | relative joint targets (q_target = q_target,prev + s·a, HORA/LEAP) | small, fast finger motions are easy to express | **failed / killed** at it 96 on rotlin + ent 5e-3 (29845929 / A10) |

Order: decide after the reward evals; `alpha05` / `relact` change the action handling that the policy node must mirror
(allowed: we fork `rwr_faive_system`, see [sim2real](sim2real.md#deploy-repo-facts-task-pdf)). EMA smoothing must be
applied at the same point in the chain in sim and deploy.

## Results

- **2026-10-29 08:40 — `relact` cannot learn from our reset** (`abl_B_rotlin_ent5e-3_relact`, killed at it 96):
  mean episode length fell from 9 to 6 steps (0.3 s), 99.98 % of episodes ended in a drop; the absolute-target rotlin
  run had 60 steps at it 20 and 369 at it 50. Cause: the relative target starts at the reset joint pose (hand near
  open, ±20 % of the limits) and moves at most 10 % of the half-range per step, so the hand cannot close before the
  cube falls, and short episodes give no learning signal. HORA and LEAP use relative targets **together with
  grasp-cache resets** (episodes start from a stable grasp), which we do not have.

## Conclusions so far

- None yet; the perturbed-eval speed-up is the main hint that action filtering matters.

## Next steps / open questions

- `alpha05` queued on rotlin + ent 5e-3.
- `relact` only with grasp-cache resets (HORA `gen_grasp.py`, LEAP_Hand_Sim cache) or a reset that starts the
  relative target at a closed grasp pose; a larger `delta_scale` (e.g. 0.3) is the cheap alternative to try first.
- Spin-command curriculum (start slow, increase): not scheduled.
