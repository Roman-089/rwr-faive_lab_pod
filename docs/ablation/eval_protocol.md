# Eval protocol

[Index](README.md) · metrics come from the ORCA PT paper (arXiv 2605.21330, see [literature](literature.md#orca-proprioceptive-transformer-paper-takeaways)).

## Why not training reward

Reward is not comparable across variants (different obs noise / DR / reward terms), so every run is scored with the
paper's metrics on a **fixed eval setting**. Mean episode reward and episode length from the training logs are kept for
reference only.

## Metrics (from the paper; use for every run, sim and real)

Fixed eval setting: command 1.5 rad/s about z (fixed sign), 60 s episodes (1200 policy steps at 20 Hz), DR + obs
noise **on** (the training distribution of the baseline), same seed.

| Metric | Definition (ours, sim) | Paper (real, PT-Motor) |
|---|---|---|
| **RPM** | cube rotation in the commanded direction per minute: ∫ sign(cmd)·ω_z dt / 2π over the whole 60 s trial (dropped cubes re-spawn and keep counting), mean over envs | 9.33 |
| **Rotation accuracy (RA)** | commanded-direction rotation / total absolute rotation (∫ max(0, sign(cmd)·ω_z) / ∫ \|ω_z\|), mean over envs. The paper's exact definition may differ (not checked against its §IV) | 96.2 % |
| **Drop-free success rate (DFSR)** | % of 60 s trials without a drop | 98.1 % |
| **Drop count (DC)** | drops per 60 s trial (cube re-spawned after a drop, keep counting) | 0.33 |

- Implemented: one eval of 64 envs × 60 s per checkpoint, command sign alternating over envs (an earlier plan said 3 seeds). Real: 3 × 60 s trials, as in the paper.
- Also log RPM / DFSR at a lower command (0.9 rad/s) and with direction flips (deploy node's `goal_flip_period_s`).
- |cmd| 1.5 rad/s = 14.3 RPM. Training commands sample |ω*| in 60–100 % of 1.5 rad/s about ±z, resampled every 8 s,
  so the eval's 1.5 rad/s is the top of the training range ([rewards](rewards.md)).

Definitions above are the implemented ones (`scripts/eval/eval_policy.py`, `metrics()`).

## Nominal and perturbed eval

Each run is scored twice, columns are always "nominal / perturbed":

- **Nominal:** the task's own DR and obs noise.
- **Perturbed** (`eval_policy.py --perturb`): same policy with 50–100 ms action delay (6–12 physics substeps) and a
  per-episode ±0.05 rad joint obs offset at test time only, as a sim proxy for robustness to unmodeled effects
  (e.g. whether a history-based policy degrades less than a single-observation MLP).

Per-env RPM std is ~1.5, so the SE over 64 envs is ~0.2 (≈0.18).

## Checkpoint selection

- Runs save every 500 iters. `hpc/eval_run.sh` scores 5 checkpoints: ~1/5, 2/5, 3/5, 4/5 of training + the last one
  (sorted with `sort -V`; see the sort bug in [infrastructure](infrastructure.md#storage-quota-and-pruning)).
- A's eval *drops* after it ~4.5k while the training reward plateaus → **report the best checkpoint per run, not only
  the last** (results tables give "last (best @iter)").
- A was scored at 2500/4500/6250/8000/9999 instead of evenly spaced ones (last = 9999); not invalidated by the bug.

## End-of-training eval pipeline

`train_job_ablation.sh` ends with `bash hpc/eval_run.sh RUN_DIR TASK` (~15 min, keep it in the walltime):

1. `scripts/eval/eval_policy.py --num_envs 64 --steps 1200` on the 5 checkpoints (one Isaac Sim start) → `eval/metrics.json`
   and `<ckpt>.npz` rollouts.
2. Same with `--perturb` → `metrics_perturbed.json`.
3. MuJoCo videos from the rollouts (`scripts/replay/render_rollout.py`, EGL / osmesa; Isaac RTX rendering crashes on the
   cluster, see [infrastructure](infrastructure.md)).
4. `scripts/eval/log_to_wandb.py` uploads metrics + videos.
5. Deletes perturbed rollouts, runs `hpc/prune_storage.sh`.

Separately queued evals for runs submitted with `train_job_ablation.sh` are therefore redundant (cancelled
29836987 / 29836988 / 29837304). `hpc/eval_job.sh` / `hpc/submit_eval.sh` run the same pipeline standalone (e.g. for
an eval queued with `-w ended(<train job>)`).

Originally planned implementation: `hpc/replay/record_rollout.py` already logs cube quat, angular velocity, command and
dones per step; add `hpc/eval_metrics.py` that turns a recording into these four numbers (+ `--steps 1200`, fixed
command, more envs). Eval pipeline (metrics, perturbed, MuJoCo videos, W&B) verified end to end on 2026-10-27 (A).

## Speed-tracking check

Does the policy follow the commanded speed, or just rotate as fast as it can in the commanded direction? Our best
policies average ~0.78 rad/s (7.4 RPM), below the slowest training command (0.9 rad/s), so the magnitude reward may act
as plain "go faster". `hpc/speed_sweep_job.sh` evaluates one checkpoint (nominal, 60 s, 64 envs) at
|cmd| = 0.3 / 0.6 / 0.9 / 1.2 / 2.0 rad/s → `<run>/eval/metrics_cmd<c>.json` (1.5 = the standard `metrics.json`).
Flat RPM over cmd = speed command ignored → a wider or curriculum speed range (e.g. 0.3–1.5) is the next reward/command
ablation; RPM rising with cmd = the policy tracks speed and the training range matters.

| Checkpoint | 0.3 | 0.6 | 0.9 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| `abl_B_asym_ent5e-3` model_4999 (job 29845227) | 5.11 | 6.66 | 6.80 | 6.92 | 7.33 | 7.25 |
| `teacher` model_9999 (job 29845228) | 2.28 | 6.67 | 7.26 | 7.33 | 7.44 | 7.16 |

(Commanded speeds in RPM: 2.9 / 5.7 / 8.6 / 11.5 / 14.3 / 19.1.) **`ent5e-3` ignores the commanded speed**: from 0.6 to
2.0 rad/s its RPM stays at 6.7–7.3 while the command changes by 3.3×; it rotates as fast as it can in the commanded
direction. At 0.3 rad/s (below the training range 0.9–1.5) it still turns at 5.1 RPM, 1.8× the command, and becomes
less reliable (RA 53 %, DFSR 72 %). So speed tracking never mattered: the speed term only ever said "faster", which
supports the one-direction max-speed setup (`-onedir`) for the demo, and a wider command range would not help.
The privileged teacher shows the same plateau (6.7–7.4 RPM for 0.6–2.0 rad/s); only at 0.3 rad/s does it slow down
(2.3 RPM, still drop-free 92 %), so it does react to a near-zero command but not to the speed in the trained range.

## Drop timing: are better starting grasps needed? (2026-10-28, `scripts/eval/drop_timing.py`, CPU)

Nominal rollouts of 10 policies (B, B seed1, ent5e-3, C, teacher, RMA s1/s2, hist3/10, LSTM; 64 envs × 60 s each):

- **Half the drops are reset artifacts.** 51 of 99 drops come within 5 s of a (re)spawn, most at steps 7–13 (0.35–0.65 s):
  the cube slides off before the EMA-filtered fingers (~250 ms time constant) can close. Drop rate in the first second
  ≈ 2.7 per env-minute vs ≈ 0.1 in steady state (> 20×). Overall 6.9 % of spawns drop within 5 s.
- **They barely cost speed:** spin-up + early respawns cost 0.2–2.6 % of the rotation (RPM).
- **They dominate the drop-free rate of the best policies:** ent5e-3 has 6 of its 8 drops within 5 s of spawn (teacher
  3 of 4); without them, its DFSR would be ~97 % instead of 88 %. Steady-state drops of ent5e-3 / teacher: ~0.03 per
  env-minute.
- **Not relevant on the real hand:** a demo starts once, from a cube a person places in the hand, not from the sim's
  "cube dropped onto randomized fingers" reset.

**Verdict:** a grasp cache (HORA / LEAP) would mostly fix a sim-only failure. It is not where speed or real-robot
endurance is lost; deprioritized. Cheaper: report a steady-state DFSR (ignoring drops in the first ~1 s after a spawn)
so the sim metric matches the demo, and revisit a gentler reset only if training-time drops stay high.
