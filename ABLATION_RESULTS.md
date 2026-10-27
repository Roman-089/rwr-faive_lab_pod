# Ablation results — orca-v2-inhand-repose (Subtask 1, proprio only)

Plan, hypotheses and literature: `HYPERPARAM_NOTES.md`. This file only logs runs and measured results.

Common setup unless noted: 8192 envs, seed 42, baseline PPO cfg (`agents/rsl_rl_ppo_cfg.py`), W&B project
`rwr-fast/orca_v2_repose`, logs in `logs/rsl_rl/orca_v2_repose/<date>_<run_name>/`.
Submit: `hpc/submit_ablation.sh TASK RUN_NAME ITERS QUEUE WALLTIME` (script not in git; see HYPERPARAM_NOTES).

Metrics (HYPERPARAM_NOTES §8): RPM, rotation accuracy (RA), drop-free success rate (DFSR), drop count (DC) on the
fixed eval setting; mean episode reward and episode length from training logs for reference only (not comparable
across variants with different rewards / noise).

## Runs

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status | Reward (last 100 it) | Ep. len | Drop % | RPM | RA | DFSR | DC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A `firstrun_l40s` | `orca-v2-inhand-repose-v0` | baseline | 29823035 / L40S | 10k | done (2026-10-27) | ~493 | ~395 / 640 | ~62 % | | | | |
| B `abl_B_asym` | `-asym-v0` | asymmetric critic (teacher obs) | 29829722 / L40S | 5k | running | | | | | | | |
| C `abl_C_asym_dr` | `-asym-dr-v0` | B + cube vel + DR params in critic | 29829723 / A40 | 5k | running | | | | | | | |
| sanity `sanity_nodr` | `-nodr-v0` | no DR, no obs noise | 29829724 / A10 | 5k | running | | | | | | | |

## Observations / log

- 2026-10-27: A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
