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
| B `abl_B_asym` | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | 5k | done (2026-10-27 18:15) | ~1573 | ~588 / 640 | ~10 % | 5.66 / 6.55 | 66.3 / 62.0 | 86 / 52 | 0.20 / 0.98 |
| C `abl_C_asym_dr` | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | 5k | done (2026-10-27 22:35) | ~1583 | ~582 / 640 | ~12 % | 6.08 / 6.40 (best 6.19 / 6.77 @4000) | 72.2 / 63.2 | 84 / 69 | 0.20 / 0.44 |
| sanity `sanity_nodr` | `-nodr-v0` | no DR, no obs noise | 29829724 / A10 | 5k | done (2026-10-27 22:45) | ~1793 | ~574 / 640 | ~10 % | 10.55 / 10.59 | 85.8 / 85.7 | 83 / 69 | 0.19 / 0.47 |
| `arch_asym_hist3` | `-asym-hist3-v0` | B + MLP on 3-step obs history | 29831099 / A10 | 5k | running | | | | | | | |
| `arch_asym_hist10` | `-asym-hist10-v0` | B + MLP on 10-step obs history | 29831100 / A40 | 5k | queued | | | | | | | |
| `arch_asym_lstm` | `-asym-lstm-v0` | B + LSTM actor-critic (PPO) | 29831098 / A100 | 5k | done (2026-10-28 06:40) | ~1437 | | ~13 % | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
| `abl_B_asym_seed1` | `-asym-v0` | B, seed 1 (replicate of the main effect) | 29832230 / L40S | 5k | done (2026-10-28 06:40) | ~1568 | | ~12 % | 5.23 / 5.92 (best 5.48 / 6.16 @2000) | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
| `abl_B_asym_ent5e-3` | `-asym-v0` | B + entropy_coef 1e-4 → 5e-3 (hydra override) | 29833758 / A100 | 5k | running | | | | | | | |
| `arch_rma_s1` | `-rma-teacher-v0` | B + RMA extrinsics encoder in the actor (HORA stage 1; not deployable) | 29835277 / L40S | 5k | done (2026-10-28 07:30) | ~1629 | | ~10 % | 6.48 / 6.63 (best 6.55 / 7.03 @4000) | 74.5 / 69.3 | 89 / 66 | 0.16 / 0.61 |
| `arch_rma_s2` | `-rma-student-v0` | RMA stage 2: adaptation module on 30-step history (deployable) | 29837730 / L40S | 2k | queued | | | | | | | |
| `abl_B_droppen` | `-asym-droppen-v0` | B + −100 return on drop (AnyRotate / Aero Hand) | 29835278 / A40 | 5k | queued | | | | | | | |
| `abl_C_asym_dr_seed1` | `-asym-dr-v0` | C, seed 1 (is C > B real?) | 29838635 / A10 | 5k | queued | | | | | | | |
| `abl_B_rotlin` | `-asym-rotlin-v0` | B, rotation reward linear in speed, clipped at the command (HORA/LEAP/DLR 2024) + off-axis penalty | 29838450 / L40S | 5k | queued | | | | | | | |
| `abl_B_rotsoft` | `-asym-rotsoft-v0` | B, magnitude kernel σ 0.2 → 0.75, weights mag 30 → 60 / dir 60 → 30 | 29838451 / A100 | 5k | queued | | | | | | | |
| `student_mlp_h10` | `-student-mlp-h10-v0` | distill teacher → MLP on 10-step noisy proprio history (PT paper's MLP student) | 29839014 / A100 | 2k | queued after teacher | | | | | | | |
| `student_lstm` | `-student-lstm-v0` | distill teacher → LSTM student | 29839015 / A40 | 2k | queued after teacher | | | | | | | |
| `student_pt` | `-student-pt-v0` | distill teacher → transformer on 10-step history + reconstruction loss (PT paper) | 29839013 / L40S | 2k | queued after teacher | | | | | | | |
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

- 2026-10-28 02:00 — **drop timing (A, model_9999, nominal eval, 116 drops):** time since spawn 0–2 s 12 %, 2–5 s 15 %,
  5–10 s 16 %, 10–20 s 23 %, 20–40 s 22 %, 40–61 s 12 %. Drops happen throughout the episode, not mainly while the cube
  settles after the reset, so **grasp-cache resets (HORA/LEAP) are deprioritized** (they target initial-grasp
  failures); drop penalty and entropy runs address sustained-grip drops. Re-check on B's rollout once its eval runs.

- 2026-10-28 01:50 — **storage:** home quota 25.6 of 30 GB. `hpc/prune_storage.sh` keeps finished runs to their 5
  eval checkpoints (+ any already scored), drops redundant eval rollouts and synced local W&B logs; runs after every
  eval. First pass freed 1.06 GB. New runs save every 500 iters. **Bug found and fixed:** `eval_run.sh` sorted
  checkpoints as text (the run-dir underscores broke `sort -t_ -k2 -n`), so a 5k run's "last" checkpoint would have
  been `model_750.pt`; now `sort -V`. A's eval was not invalidated (it scored 2500/4500/6250/8000/9999 instead of
  evenly spaced ones, last = 9999); B/C/sanity evals had not started yet.

- 2026-10-28 04:00 — **sanity eval (no DR / no noise, evaluated on its own no-DR task)**, nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 8.92 / 9.09 | 76.6 / 75.1 | 86 / 64 | 0.14 / 0.55 |
  | 2000 | 9.32 / 9.41 | 77.8 / 78.0 | 72 / 67 | 0.34 / 0.52 |
  | 3000 | 9.85 / 9.25 | 82.6 / 76.6 | 89 / 66 | 0.11 / 0.53 |
  | 4000 | 9.43 / 9.62 | 80.2 / 77.5 | 88 / 53 | 0.14 / 0.75 |
  | 4999 | **10.55 / 10.59** | **85.8 / 85.7** | 83 / 69 | 0.19 / 0.47 |

  |cmd| 1.5 rad/s = 14.3 RPM, so the policy tracks ~74 % of the commanded speed; the perturbation (delay + joint
  offset) does not slow it but roughly triples drops. ~8× A's best RPM — but **not comparable to A** (evaluated without
  DR / noise). It is the upper reference for what the reward / action space allow; the gap to it is what DR + obs
  noise cost. To separate "DR makes the task harder" from "DR makes learning harder", the sanity `model_4999` is also
  evaluated on the DR task (`sanity_nodr_on_dr`, job 29836428).

- 2026-10-28 06:45 — **evals: asymmetric critic family** (last checkpoint, nominal / perturbed, same protocol as A):

  | Run | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | A baseline (best @4500) | 1.37 / 1.62 | 19.1 / 19.0 | 33 / 25 | 1.3 / 2.0 |
  | B seed 1 | 5.23 / 5.92 | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
  | C (+ cube vel + DR params in critic) | **6.08 / 6.40** | **72.2 / 63.2** | 84 / 69 | 0.20 / 0.44 |
  | asym LSTM | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
  | sanity, no DR (own no-DR env, reference) | 10.55 / 10.59 | 85.8 / 85.7 | 83 / 69 | 0.19 / 0.47 |

  1. **The asymmetric critic is the dominant effect: ~4× A's best RPM, 3× the rotation accuracy, ~3× the drop-free
     rate** (B seed 1; B seed 42's eval is still queued). Matches the training-reward picture (1570 vs 490).
  2. **C beats B in eval (6.1 vs 5.2 RPM, RA 72 vs 63 %) although their training rewards are equal** — but B is a
     different seed here; decide once B seed 42 is scored. If it holds, DR params in the critic help the speed /
     accuracy terms that the reward weighs weakly.
  3. **LSTM actor-critic ≈ B** nominal; slightly better under perturbation (DC 0.70 vs 0.86) but within seed noise.
     Memory alone does not close the gap at 5k iters (consistent with its lower training reward).
  4. **Every policy rotates faster under the perturbation** (A, B, C, LSTM: +0.3 to +1 RPM) while dropping 3–8× more.
     The 1–2-step delay apparently acts as extra low-pass / phase lag on finger gaiting. Recorded; action smoothing is a
     control choice, not part of the Phase 1 architecture study.
  5. Gap to the no-DR reference is ~2× in RPM: DR + obs noise cost half the speed. `sanity_nodr_on_dr` (queued)
     separates "DR makes the task harder" from "DR makes learning harder".

  Eval note: `train_job_ablation.sh` already ends with `hpc/eval_run.sh`, so separately queued evals for runs submitted
  with it are redundant (cancelled 29836987 / 29836988 / 29837304).

- 2026-10-28 07:30 — **jobs hung after finishing:** B seed 1 and asym LSTM wrote their last output (eval, videos, W&B,
  prune) at 06:14 / 05:58 but stayed RUN for over an hour, holding an L40S and an A100. LSF showed one leftover 3-thread
  64 MB process in each job's process group (most likely an orphaned W&B service from `train.py`). Both killed (all
  outputs were saved); `train_job_ablation.sh` now kills leftovers in its own process group and exits 0. Jobs submitted
  before the fix are caught by the watcher (`[prune]` line in the log while still RUN) and killed by hand; job
  dependencies use `ended()` rather than `done()` since a killed job ends as EXIT.

- 2026-10-28 07:35 — **RMA stage 1 eval** (actor sees cube pos + DR params through an 8-d latent; upper bound for the
  deployable stage 2): 6.48 / 6.63 RPM, RA 74.5 / 69.3 %, DFSR 89 / 66 %. Only ~+0.4 RPM over C (6.08) and +1.3 over
  B seed 1 (5.23), although its training reward is clearly higher (1629 vs ~1570). **Knowing the physics parameters
  explicitly buys little**, so the remaining gap to the no-DR reference (10.6 RPM) is not mainly about identifying the
  DR sample — more likely obs noise and the harder dynamics themselves. Stage 2 can at best recover these ~6.5 RPM.

- 2026-10-28 09:30 — **reward review** (HYPERPARAM_NOTES §14): our direction term pays the same for any speed and the
  magnitude kernel is ~0 at our policies' speed, so speed is barely rewarded; the literature rewards speed linearly and
  clips it. Queued `abl_B_rotlin` (linear clipped progress) and `abl_B_rotsoft` (reweighted current terms).

- 2026-10-28 09:35 — **B (seed 42) eval:** 5.66 / 6.55 RPM, RA 66.3 / 62.0 %, DFSR 86 / 52 %. Two seeds of B: 5.23 and
  5.66 RPM (mean 5.45, spread ±0.2; per-run SE over 64 envs ~0.18). C at the same seed: 6.08 RPM, RA 72 vs 66 % →
  **C is +0.4 RPM / +6 pp RA over B at matched seed, about 2× the seed spread**: a small, plausible gain from DR params
  + cube velocity in the critic, not yet conclusive (one C seed). RMA stage 1 (6.48) is ~+0.8 over B. The asymmetric
  critic itself (A 1.4 → B 5.5) remains the only large architectural effect so far.

- 2026-10-27: gpuv100 cannot run our stack (torch has no sm_70 kernels) → 4 usable queues: gpua100, gpua40, gpua10, gpul40s.
- 2026-10-27: A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
