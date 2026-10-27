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
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline | 29823035 / L40S | 10k | done (2026-10-27) | ~493 | ~395 / 640 | ~62 % | | | | |
| B `abl_B_asym` | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | 5k | running | | | | | | | |
| C `abl_C_asym_dr` | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | 5k | running | | | | | | | |
| sanity `sanity_nodr` | `-nodr-v0` | no DR, no obs noise | 29829724 / A10 | 5k | running | | | | | | | |
| `arch_asym_hist3` | `-asym-hist3-v0` | B + MLP on 3-step obs history | 29831099 / A10 | 5k | queued | | | | | | | |
| `arch_asym_hist10` | `-asym-hist10-v0` | B + MLP on 10-step obs history | 29831100 / A40 | 5k | queued | | | | | | | |
| `arch_asym_lstm` | `-asym-lstm-v0` | B + LSTM actor-critic (PPO) | 29831098 / A100 | 5k | queued | | | | | | | |
| `abl_B_asym_seed1` | `-asym-v0` | B, seed 1 (replicate of the main effect) | 29832230 / L40S | 5k | queued | | | | | | | |
| `teacher` | `-teacher-v0` | privileged teacher (sim only) | 29830926 / L40S | 10k | queued | | | | | | | |

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

- 2026-10-27: gpuv100 cannot run our stack (torch has no sm_70 kernels) → 4 usable queues: gpua100, gpua40, gpua10, gpul40s.
- 2026-10-27: A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
