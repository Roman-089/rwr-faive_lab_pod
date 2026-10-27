# Ablation results — orca-v2-inhand-repose (Subtask 1, proprio only)

Plan, hypotheses and literature: `HYPERPARAM_NOTES.md`. This file only logs runs and measured results.

Common setup unless noted: 8192 envs, seed 42, baseline PPO cfg (`agents/rsl_rl_ppo_cfg.py`), W&B project
`rwr-fast/orca_v2_repose`, logs in `logs/rsl_rl/orca_v2_repose/<date>_<run_name>/`.
Submit: `hpc/submit_ablation.sh TASK RUN_NAME ITERS QUEUE WALLTIME` (script not in git; see HYPERPARAM_NOTES).

Metrics (HYPERPARAM_NOTES §8): RPM, rotation accuracy (RA), drop-free success rate (DFSR), drop count (DC) on the
fixed eval setting; mean episode reward and episode length from training logs for reference only (not comparable
across variants with different rewards / noise).

Phase 1 = architecture study on the baseline env (see HYPERPARAM_NOTES "Run plan"). Each run: nominal eval and
perturbed eval (test-time 50–100 ms action delay + ±0.05 rad joint obs offset), columns "nominal / perturbed".

## Runs

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status | Reward (last 100 it) | Ep. len | Drop % | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline | 29823035 / L40S | 10k | done (2026-10-27) | ~493 | ~395 / 640 | ~62 % | 0.61 / 1.16 (best 1.37 / 1.62 @4500) | 11.6 / 14.6 (19.1 / 19.0) | 22 / 20 (33 / 25) | 1.8 / 2.2 (1.3 / 2.0) |
| B `abl_B_asym` | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | 5k | done (2026-10-27 18:15) | ~1573 | ~588 / 640 | ~10 % | | | | |
| C `abl_C_asym_dr` | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | 5k | done (2026-10-27 22:35) | ~1583 | ~582 / 640 | ~12 % | | | | |
| sanity `sanity_nodr` | `-nodr-v0` | no DR, no obs noise | 29829724 / A10 | 5k | done (2026-10-27 22:45) | ~1793 | ~574 / 640 | ~10 % | | | | |
| `arch_asym_hist3` | `-asym-hist3-v0` | B + MLP on 3-step obs history | 29831099 / A10 | 5k | queued | | | | | | | |
| `arch_asym_hist10` | `-asym-hist10-v0` | B + MLP on 10-step obs history | 29831100 / A40 | 5k | queued | | | | | | | |
| `arch_asym_lstm` | `-asym-lstm-v0` | B + LSTM actor-critic (PPO) | 29831098 / A100 | 5k | running | | | | | | | |
| `abl_B_asym_seed1` | `-asym-v0` | B, seed 1 (replicate of the main effect) | 29832230 / L40S | 5k | running | | | | | | | |
| `abl_B_asym_ent5e-3` | `-asym-v0` | B + entropy_coef 1e-4 → 5e-3 (hydra override) | 29833758 / A100 | 5k | queued | | | | | | | |
| `arch_rma_s1` | `-rma-teacher-v0` | B + RMA extrinsics encoder in the actor (HORA stage 1; not deployable) | 29835277 / L40S | 5k | queued | | | | | | | |
| `arch_rma_s2` | `-rma-student-v0` | RMA stage 2: adaptation module on 30-step history (deployable) | after s1 | 2k | planned | | | | | | | |
| `abl_B_droppen` | `-asym-droppen-v0` | B + −100 return on drop (AnyRotate / Aero Hand) | 29835278 / A40 | 5k | queued | | | | | | | |
| `teacher` | `-teacher-v0` | privileged teacher (sim only) | 29830926 / L40S | 10k | running (started 2026-10-27 23:55) | | | | | | | |

## Observations / log

- 2026-10-27 15:20 — **interim, matched iterations (training logs, mean of last 50 iters):**

  | Run | it 1000 rew / len / drop | it 1500 | it 2400 |
  |---|---|---|---|
  | A baseline | 307 / 318 / 0.76 | 439 / 388 / 0.64 | 447 / 374 / 0.64 |
  | B asym critic | 998 / 542 / 0.22 | 1148 / 561 / 0.19 | **1325 / 571 / 0.14** |
  | C asym + DR params | 949 / 538 / 0.24 | 1058 / 551 / 0.22 | – |
  | sanity (no DR / noise) | 1425 / 564 / 0.15 | 1585 / 576 / 0.13 | 1581 / 571 / 0.13 |

  (drop = fraction of episode ends that are drops). The asymmetric critic is by far the largest effect so far: ~3×
  reward and drops 64 % → 14 %, approaching the no-DR upper reference. C (+ cube vel + DR params) is not better than B
  yet. **Decision:** the queued history / LSTM runs were re-based on B (`-asym-hist3/-asym-hist10/-asym-lstm`); the
  A-based versions (`-hist3/-hist10/-lstm`, jobs 29830927/28/30) were cancelled before starting.

- 2026-10-27 17:20 — B 1520 / 585 / 0.12 at it 4190; C 1457 / 592 / 0.14 at it 2160 (C now *ahead* of B at
  matched iterations, B was 1325 at it 2400 — the C gap at it 1500 was transient); sanity 1742 / 578 / 0.11 at it
  2490. Evals for B / C / sanity queued with `-w ended(<train job>)` (jobs 29832523 / 29832525 / 29832526).

- 2026-10-27 19:45 — **first eval (A, baseline)**, nominal / perturbed, 64 envs × 60 s, |cmd| 1.5 rad/s:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 2500 | 1.25 / 1.31 | 17.4 / 17.3 | 36 / 12.5 | 1.6 / 2.7 |
  | 4500 | **1.37 / 1.62** | **19.1 / 19.0** | 33 / 25 | 1.3 / 2.0 |
  | 6250 | 0.65 / 1.29 | 12.6 / 15.4 | 14 / 14 | 2.1 / 2.9 |
  | 8000 | 0.72 / 1.15 | 10.7 / 13.7 | 12.5 / 4.7 | 2.0 / 3.2 |
  | 9999 | 0.61 / 1.16 | 11.6 / 14.6 | 22 / 20 | 1.8 / 2.2 |

  A's eval performance *drops* after it ~4.5k while the training reward plateaus (action std collapses to 0.18) →
  report the best checkpoint per run, not only the last. Far below the PT paper's Proprio-PPO (3.83 RPM). Late
  checkpoints rotate *faster* with the test-time delay (0.6 → 1.2 RPM; per-env RPM std ~1.5, so SE ~0.2): hint that
  the late policy is jittery and a low-pass (delay) helps — worth checking action-rate / EMA alpha later.
  Videos + metrics on W&B run `ffueb4yh`. Eval pipeline (metrics, perturbed, MuJoCo videos, W&B) verified end to end.

- 2026-10-27 20:25 — queued `abl_B_asym_ent5e-3`: A's eval collapses after the action std shrinks (1.5 → 0.18) while
  the training reward stays flat; our entropy_coef 1e-4 is 100× below legged_gym's 1e-2. Hypothesis: more entropy
  keeps exploration up and avoids the late eval drop. B's eval moved to A100 (L40S hosts full for >2 h).

- 2026-10-28 01:30 — literature → runs (HYPERPARAM_NOTES §13): added **RMA** (HORA's public MIT code, adaptation
  module ported 1:1; CPU unit test: PPO trains the env encoder, stage-1 checkpoint loads into stage 2 with identical
  teacher actions, distillation trains only the 24 k-param adaptation module) and **B + drop penalty**. Grasp-cache
  resets next (HORA / LEAP code available).

- 2026-10-27: gpuv100 cannot run our stack (torch has no sm_70 kernels) → 4 usable queues: gpua100, gpua40, gpua10, gpul40s.
- 2026-10-27: A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
