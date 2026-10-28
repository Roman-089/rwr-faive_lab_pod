# Asymmetric critic (A baseline, B, C, seeds)

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

Does giving the critic privileged, noise-free state (asymmetric actor-critic) fix the weak baseline, while the deployed
actor keeps the same 74-dim noisy obs? Does adding cube velocity + the sampled DR parameters to the critic (C) help
further?

## Literature

- Pinto et al. 2017 (1710.06542), also Faive, DeXtreme, Dactyl: asymmetric critic (standard, no code needed).
- C: RMA / HORA (DR params as privileged info), DeXtreme critic.
- PT paper (2605.21330): Proprio-PPO (= our run A) 3.83 RPM on hardware. See [literature](literature.md).

## Baseline A

`firstrun_l40s` (W&B `rwr-fast/orca_v2_repose`, run `ffueb4yh`), 10k iters, 8192 envs, L40S ~4.3 s/iter (~12 h).
Reward plateaued at ~490–510 from iter ~4000; ~60 % of episodes end in a cube drop; mean episode length ~395 / 640 steps.
Action noise std collapses to 0.18 (from 1.5). Proposed changes for A, from the original notes:

| Idea | Change | Notes |
|---|---|---|
| **Asymmetric critic** (try first) | `critic: TeacherObsGroupCfg = TeacherObsGroupCfg()` in `ObservationsCfg` | critic sees noise-free cube pose; deployed actor unchanged |
| Add cube lin/ang vel to critic | uncomment in `TeacherObsGroupCfg` | |

## Runs

Common: seed 42, 8192 envs, baseline PPO cfg, compare with A **at the same iteration** (A's curve to 5k is in W&B).
Training columns: reward = mean of last 100 iters.

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) | Reward | Ep. len | Drop % |
|---|---|---|---|---|---|---|---|---|
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline (critic = policy obs) | 29823035 / L40S | 10k | done (2026-10-27) | ~493 | ~395 / 640 | ~62 % |
| B `abl_B_asym` | `-asym-v0` | critic: noise-free joints + cube pose (`TeacherObsGroupCfg`) | 29829722 / L40S | 5k | done (2026-10-27 18:15) | ~1573 | ~588 / 640 | ~10 % |
| C `abl_C_asym_dr` | `-asym-dr-v0` | B + cube lin/ang vel + sampled DR params (mass, friction, gain scales, gravity) | 29829723 / A40 | 5k | done (2026-10-27 22:35) | ~1583 | ~582 / 640 | ~12 % |
| `abl_B_asym_seed1` | `-asym-v0` | B, seed 1 (replicate of the main effect) | 29832230 / L40S | 5k | done (2026-10-28 06:40) | ~1568 | | ~12 % |
| `abl_C_asym_dr_seed1` | `-asym-dr-v0` | C, seed 1 (is C > B real?) | 29838635 / A10 | 5k | queued | | | |

Run-plan summaries (training logs):
- A: reward ~493 (last 100 iters), drops ~62 %.
- B at 5k (train, last 50 iters): reward 1573 vs A 477; drops 10 % vs A 65 %; ep. length 588 vs 374 / 640;
  action-rate penalty −0.76 vs A −0.28 (busier fingers, check smoothness).
- C at 5k (train, iters 4900–4999): reward 1584 vs B 1568; drops 12 % vs B 10 %; ep. length 583. Ahead of B mid-run
  (1390 vs 1288 @2k) but ≈ B at the end — within single-seed noise, extra critic inputs not clearly worth it (in training).

## Results

### Eval (nominal / perturbed)

| Run | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| A, last (9999) | 0.61 / 1.16 | 11.6 / 14.6 | 22 / 20 | 1.8 / 2.2 |
| A, best @4500 | 1.37 / 1.62 | 19.1 / 19.0 | 33 / 25 | 1.3 / 2.0 |
| B seed 42 | 5.66 / 6.55 | 66.3 / 62.0 | 86 / 52 | 0.20 / 0.98 |
| B seed 1 (best 5.48 / 6.16 @2000) | 5.23 / 5.92 | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
| C (best 6.19 / 6.77 @4000) | 6.08 / 6.40 | 72.2 / 63.2 | 84 / 69 | 0.20 / 0.44 |

### Log

- **2026-10-27 15:20 — interim, matched iterations** (training logs, mean of last 50 iters; drop = fraction of episode
  ends that are drops):

  | Run | it 1000 rew / len / drop | it 1500 | it 2400 |
  |---|---|---|---|
  | A baseline | 307 / 318 / 0.76 | 439 / 388 / 0.64 | 447 / 374 / 0.64 |
  | B asym critic | 998 / 542 / 0.22 | 1148 / 561 / 0.19 | **1325 / 571 / 0.14** |
  | C asym + DR params | 949 / 538 / 0.24 | 1058 / 551 / 0.22 | – |
  | sanity (no DR / noise) | 1425 / 564 / 0.15 | 1585 / 576 / 0.13 | 1581 / 571 / 0.13 |

  The asymmetric critic is by far the largest effect so far: ~3× reward and drops 64 % → 14 %, approaching the no-DR
  upper reference. C (+ cube vel + DR params) is not better than B yet. **Decision:** the queued history / LSTM runs
  were re-based on B (see [history and memory](history_and_memory.md)).

- **2026-10-27 17:20** — B 1520 / 585 / 0.12 at it 4190; C 1457 / 592 / 0.14 at it 2160 (C now *ahead* of B at matched
  iterations, B was 1325 at it 2400 — the C gap at it 1500 was transient); sanity 1742 / 578 / 0.11 at it 2490. Evals
  for B / C / sanity queued with `-w ended(<train job>)` (jobs 29832523 / 29832525 / 29832526).

- **2026-10-27 19:45 — first eval (A, baseline)**, nominal / perturbed, 64 envs × 60 s, |cmd| 1.5 rad/s:

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
  the late policy is jittery and a low-pass (delay) helps — worth checking action-rate / EMA alpha later
  ([action space](action_space.md)). Videos + metrics on W&B run `ffueb4yh`. Eval pipeline (metrics, perturbed, MuJoCo
  videos, W&B) verified end to end. (Entropy follow-up: [PPO and capacity](ppo_and_capacity.md).)

- **2026-10-28 06:45 — evals: asymmetric critic family** (last checkpoint, nominal / perturbed, same protocol as A):

  | Run | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | A baseline (best @4500) | 1.37 / 1.62 | 19.1 / 19.0 | 33 / 25 | 1.3 / 2.0 |
  | B seed 1 | 5.23 / 5.92 | 62.6 / 61.1 | 91 / 55 | 0.11 / 0.86 |
  | C (+ cube vel + DR params in critic) | **6.08 / 6.40** | **72.2 / 63.2** | 84 / 69 | 0.20 / 0.44 |
  | asym LSTM | 5.12 / 6.15 | 63.9 / 63.2 | 91 / 59 | 0.09 / 0.70 |
  | sanity, no DR (own no-DR env, reference) | 10.55 / 10.59 | 85.8 / 85.7 | 83 / 69 | 0.19 / 0.47 |

  1. **The asymmetric critic is the dominant effect: ~4× A's best RPM, 3× the rotation accuracy, ~3× the drop-free
     rate** (B seed 1; B seed 42's eval was still queued). Matches the training-reward picture (1570 vs 490).
  2. **C beats B in eval (6.1 vs 5.2 RPM, RA 72 vs 63 %) although their training rewards are equal** — but B is a
     different seed here; decide once B seed 42 is scored. If it holds, DR params in the critic help the speed /
     accuracy terms that the reward weighs weakly.
  3. **LSTM actor-critic ≈ B** nominal; slightly better under perturbation (DC 0.70 vs 0.86) but within seed noise.
     Memory alone does not close the gap at 5k iters (consistent with its lower training reward). → [history and memory](history_and_memory.md)
  4. **Every policy rotates faster under the perturbation** (A, B, C, LSTM: +0.3 to +1 RPM) while dropping 3–8× more.
     The 1–2-step delay apparently acts as extra low-pass / phase lag on finger gaiting. Recorded; action smoothing is a
     control choice, not part of the Phase 1 architecture study ([action space](action_space.md)).
  5. Gap to the no-DR reference is ~2× in RPM: DR + obs noise cost half the speed. `sanity_nodr_on_dr` (queued)
     separates "DR makes the task harder" from "DR makes learning harder" ([sim2real](sim2real.md#sanity-runs-no-dr)).

  Eval note: `train_job_ablation.sh` already ends with `hpc/eval_run.sh`, so separately queued evals for runs submitted
  with it are redundant (cancelled 29836987 / 29836988 / 29837304).

- **2026-10-28 09:35 — B (seed 42) eval:** 5.66 / 6.55 RPM, RA 66.3 / 62.0 %, DFSR 86 / 52 %. Two seeds of B: 5.23 and
  5.66 RPM (mean 5.45, spread ±0.2; per-run SE over 64 envs ~0.18). C at the same seed: 6.08 RPM, RA 72 vs 66 % →
  **C is +0.4 RPM / +6 pp RA over B at matched seed, about 2× the seed spread**: a small, plausible gain from DR params
  + cube velocity in the critic, not yet conclusive (one C seed). RMA stage 1 (6.48) is ~+0.8 over B ([RMA](rma.md)).
  The asymmetric critic itself (A 1.4 → B 5.5) remains the only large architectural effect so far.

## Conclusions so far

- Asymmetric critic: A 1.4 (best) → B 5.45 RPM (two-seed mean), the only large architectural effect so far. B is the
  base of the history, LSTM, RMA, reward and entropy runs.
- C vs B: +0.4 RPM / +6 pp RA at matched seed, ~2× the seed spread; not conclusive with one C seed.
- A is the weak baseline: eval peaks at it ~4.5k and then degrades as action std collapses.

## Next steps / open questions

- `abl_C_asym_dr_seed1` decides whether C > B is real.
- Whether the late-checkpoint degradation of A also appears in B/C (B seed 1 best was @2000: 5.48 / 6.16).
- B's action-rate penalty (−0.76 vs −0.28) → check smoothness on hardware.
