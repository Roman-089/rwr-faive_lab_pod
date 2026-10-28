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
| `alpha05` | EMA α 0.2 → 0.5 | α 0.2 at 20 Hz is a ~5-step (≈250 ms) time constant on joint targets; may cap gaiting speed | planned |
| `relact` | relative joint targets (q_target = q_target,prev + s·a, HORA/LEAP) | small, fast finger motions are easy to express | planned |

Order: decide after the reward evals; `alpha05` / `relact` change the action handling that the policy node must mirror
(allowed: we fork `rwr_faive_system`, see [sim2real](sim2real.md#deploy-repo-facts-task-pdf)). EMA smoothing must be
applied at the same point in the chain in sim and deploy.

## Results

None yet.

## Conclusions so far

- None yet; the perturbed-eval speed-up is the main hint that action filtering matters.

## Next steps / open questions

- Run `alpha05` / `relact` on the reward-wave winner.
- Spin-command curriculum (start slow, increase): not scheduled.
