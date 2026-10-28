# RMA (HORA port): stage 1 and stage 2

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

RMA as a cheaper alternative to the transformer student (ranked idea 7 in the
[in-hand rotation review](literature.md#in-hand-rotation-literature-review)): stage 1 trains B with an actor that also
sees an 8-d latent of the privileged physics (cube pos + DR params); stage 2 trains an adaptation module that estimates
that latent from a 30-step proprio history, which makes the policy deployable. Stage 1 is the upper bound for stage 2.

## Literature

- HORA 2210.04887: 3-step joint pos + targets; RMA latent (8-d) from 30-step 1D-conv adaptation; real TTF 0.98 vs 0.72
  plain DR. MLP with long raw history collapses (OOD rot 141 @3 steps → 36 @30).
- **Code: yes, MIT**: [HaozhiQi/hora](https://github.com/HaozhiQi/hora). `ProprioAdaptTConv` ported 1:1 into
  `tasks/orca_inhand/rma.py` (30 steps, conv 9/2-5-5, 8-d tanh latent, MSE on latent).
- HORA details that matter for us (from the code): reward values in the repo are higher than the paper because the reset
  order follows legged_gym; their adaptation module's comment says 50 steps but the conv stack only fits 30.
- General context: [literature](literature.md).

## Runs

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Drop % |
|---|---|---|---|---|---|---|---|
| `arch_rma_s1` | `-rma-teacher-v0` | B + RMA extrinsics encoder in the actor (HORA stage 1; actor sees tanh(env_mlp(cube pos, DR params)); not deployable) | 29835277 / L40S | 5k | done (2026-10-28 07:30) | ~1629 | ~10 % |
| `arch_rma_s2` | `-rma-student-v0` | RMA stage 2: adaptation module on 30-step history (deployable) | 29837730 / L40S | 2k | done (2026-10-28 17:10) | | |

## Results (nominal / perturbed)

| Run | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| RMA s1, last (best 6.55 / 7.03 @4000) | 6.48 / 6.63 | 74.5 / 69.3 | 89 / 66 | 0.16 / 0.61 |
| RMA s2, last (best 6.29 / 6.35 @1500) | 6.20 / 6.71 | 72.3 / 68.0 | 83 / 78 | 0.25 / 0.38 |
| RMA s2 by ckpt: 0 / 500 / 1000 / 1500 / 1999 | 3.27 · 5.10 · 5.85 · 6.29 · 6.20 (nominal) | 40.6 · 61.5 · 69.3 · 73.6 · 72.3 | 59 · 72 · 80 · 86 · 83 | 0.75 · 0.39 · 0.30 · 0.17 · 0.25 |
| for comparison: C / B (two-seed mean RPM 5.45) / teacher | 6.08 · 5.45 · 7.44 (nominal) | | | |

### Log

- **2026-10-28 01:30 — literature → runs** ([paper → run mapping](literature.md#paper--run-mapping-and-code-availability)):
  added **RMA** (HORA's public MIT code, adaptation module ported 1:1; CPU unit test: PPO trains the env encoder,
  stage-1 checkpoint loads into stage 2 with identical teacher actions, distillation trains only the 24 k-param
  adaptation module) and **B + drop penalty** ([rewards](rewards.md)). Grasp-cache resets next (HORA / LEAP code available).
- **2026-10-28 07:35 — RMA stage 1 eval** (actor sees cube pos + DR params through an 8-d latent; upper bound for the
  deployable stage 2): 6.48 / 6.63 RPM, RA 74.5 / 69.3 %, DFSR 89 / 66 %. Only ~+0.4 RPM over C (6.08) and +1.3 over
  B seed 1 (5.23), although its training reward is clearly higher (1629 vs ~1570). **Knowing the physics parameters
  explicitly buys little**, so the remaining gap to the no-DR reference (10.6 RPM) is not mainly about identifying the
  DR sample — more likely obs noise and the harder dynamics themselves. Stage 2 can at best recover these ~6.5 RPM.
- 2026-10-28 09:35 — relative to B seed 42: RMA stage 1 (6.48) is ~+0.8 over B (two-seed mean 5.45).

## Conclusions so far

- Explicit knowledge of the DR sample buys only ~+0.4 RPM over C / ~+0.8–1.3 over B; the gap to the no-DR reference is
  not mainly system identification.
- **Stage 2 recovers 96 % of stage 1** (6.20 vs 6.48 RPM, RA 72 vs 75 %): the HORA adaptation module infers the
  8-d extrinsics latent from 30 steps of proprio well. It beats B (5.45) and C (6.08) as a deployable policy, and is
  among the most robust under perturbation (DFSR 78 %, DC 0.38), but is below B + entropy 5e-3 (7.33).
- Stage 2 is bounded by ~6.5 RPM, below the privileged teacher (7.44), so the
  [teacher → student](teacher_student.md) path has the higher ceiling.

## Next steps / open questions

- ~~`arch_rma_s2` eval~~: 96 % recovered.
- RMA was trained with entropy 1e-4; retrain stage 1 with entropy 5e-3 if the entropy replicate holds (stage 1 is the
  ceiling for stage 2).
- Deploying stage 2 needs the 30-step history held inside the exported policy (TorchScript wrapper).
