# Infrastructure (DTU HPC, LSF)

[Index](README.md)

> **The job scripts are not in git.** `hpc/`, `train_job_*.sh`, `activate_rwr.sh` and `run_interactive.sh` are listed
> in `.git/info/exclude`; they live only in the working copy on the cluster.

## Cluster and queues

DTU HPC uses **LSF** (`bsub`), conda env `rwr` (Isaac Lab 2.3.2 / Isaac Sim 5.1, `source activate_rwr.sh`).

| Queue | GPU | Usable | Speed (orca-v2-inhand-repose-v0, 8192 envs, no video; 2026-10-26) | Per-user limit (`blimits -u`, 2026-10-27) |
|---|---|---|---|---|
| `gpul40s` | L40S 48 GB | yes | ~4.2–4.3 s/iter (10k iters ≈ 12 h → `-W 14:00`) | 3 GPUs |
| `gpua10` | A10 24 GB | yes | ~7.5 s/iter | 2 GPUs |
| `gpua100` | A100 | yes (headless physics) | ~8.7 s/iter | by slots (72) |
| `gpua40` | A40 | yes | untested at benchmark time | 2 GPUs |
| `gpuv100` | V100 | **no**: torch has no sm_70 kernels ("no kernel image") | – | – |
| `gpuh100`, `cabl40s`, `gpukta100`, `gpudtuspace` | | restricted to other groups | – | – |

- 2026-10-27: gpuv100 cannot run our stack (torch has no sm_70 kernels) → 4 usable queues: gpua100, gpua40, gpua10, gpul40s.
- bjobs' "Job slot limit reached: N hosts" pending reason means the hosts are full with other users' jobs, not that our
  per-user limit is hit; check `blimits` first. L40S hosts were full for >2 h on 2026-10-27 evening (B's eval moved to A100).
- Fairshare is per queue and charges ≈ cores × run hours → request the 4-core minimum per GPU.
- Isaac Sim RTX rendering segfaults with the cluster driver → train `--headless` without `--video`; videos are rendered
  offline from recorded rollouts with MuJoCo (`scripts/replay/render_rollout.py`).
- Each job uses its own Isaac Sim kit root on node-local disk (`--kit_args "--portable-root $KITROOT"`), otherwise
  parallel jobs lock each other; GPU requested as `num=1:mode=shared:j_exclusive=yes` (Vulkan fails in exclusive mode).

## Submitting runs

- Variants are task ids (flags, no worktrees), defined in `tasks/orca_inhand/orca_v2_repose_ablation_env_cfg.py`.
  Same seed (42), 8192 envs and PPO cfg as the baseline unless noted. W&B project `rwr-fast/orca_v2_repose`, logs in
  `logs/rsl_rl/orca_v2_repose/<date>_<run_name>/`.
- Submit: `hpc/submit_ablation.sh TASK RUN_NAME ITERS QUEUE WALLTIME ["EXTRA train.py args"]`. DTU `bsub` does not
  forward environment variables, so the script seds the values into `train_job_ablation.sh` and pipes it to `bsub`.
  Students: `EXTRA="--load_run <teacher run dir name> --checkpoint model_9999.pt"`. Hydra overrides (e.g. entropy)
  also go into EXTRA.
- `train_job_ablation.sh`: 4 cores, 8 GB/core, default `-W 08:00`, `agent.save_interval=500`, then the
  [end-of-training eval](eval_protocol.md#end-of-training-eval-pipeline) (~15 min, keep it in the walltime).
- Chain dependent jobs with `-w "ended(ID)"`, not `done(ID)`, since a killed job ends as EXIT.
- `hpc/watch_until_event.sh JOBID...` exits as soon as one job ends or logs a crash line.
- Old first-run scripts: `train_job_<gpu>_firstrun.sh` (A ran with the L40S one, job 29823035).

## Storage quota and pruning

- **2026-10-28 01:50 — storage:** home quota 25.6 of 30 GB. `hpc/prune_storage.sh` keeps finished runs to their 5
  eval checkpoints (+ any already scored), drops redundant eval rollouts and synced local W&B logs; runs after every
  eval. First pass freed 1.06 GB. New runs save every 500 iters. **Bug found and fixed:** `eval_run.sh` sorted
  checkpoints as text (the run-dir underscores broke `sort -t_ -k2 -n`), so a 5k run's "last" checkpoint would have
  been `model_750.pt`; now `sort -V`. A's eval was not invalidated (it scored 2500/4500/6250/8000/9999 instead of
  evenly spaced ones, last = 9999); B/C/sanity evals had not started yet.
- `getquota_zhome.sh` lags up to ~4 h; `du -sh ~` for live numbers. Nothing on the cluster is backed up.

## Hanging jobs

- **2026-10-28 07:30 — jobs hung after finishing:** B seed 1 and asym LSTM wrote their last output (eval, videos, W&B,
  prune) at 06:14 / 05:58 but stayed RUN for over an hour, holding an L40S and an A100. LSF showed one leftover 3-thread
  64 MB process in each job's process group (most likely an orphaned W&B service from `train.py`). Both killed (all
  outputs were saved); `train_job_ablation.sh` now kills leftovers in its own process group and exits 0. Jobs submitted
  before the fix are caught by the watcher (`[prune]` line in the log while still RUN) and killed by hand; job
  dependencies use `ended()` rather than `done()` since a killed job ends as EXIT.

## Redundant evals

`train_job_ablation.sh` already ends with `hpc/eval_run.sh`, so separately queued evals for runs submitted with it are
redundant (cancelled 29836987 / 29836988 / 29837304, 2026-10-28 06:45). Earlier, evals for B / C / sanity were queued
with `-w ended(<train job>)` (jobs 29832523 / 29832525 / 29832526, 2026-10-27 17:20).

## Remote GPU: MicroAGI RTX 6000 Pro (2026-10-28)

Course-provided instances (RTX 6000 Pro Blackwell 96 GB, 48 vCPU, 180 GB RAM; 1 GPU per person, 8 per group, no
GPU-hour limit; reachable only over Cloudflare WARP). Uses the platform's Isaac Lab 2.3.2 image, so no Isaac Sim
install. Pull-based queue: [`scripts/remote/`](../../scripts/remote/README.md) — runs listed in
`scripts/remote/queue.txt`, status and eval metrics pushed to branch `remote-results`. Run names end in `_r6k`.
Livestreaming the sim over WebRTC also works there (platform docs), and the newer driver should allow RTX rendering
and training videos, which crash on DTU.

## RTX 6000 pod: throughput (2026-10-29)

Measured by the pod worker (8192 envs per run): 2 runs at ~5 s/it each, GPU ~80 % "util" but only ~200 of 600 W and
17 of 97 GB; a 3rd run added +3 % total and slowed the others by 30 %. The work is many small PhysX / torch kernels;
without CUDA MPS the GPU time-slices between process contexts, so kernels of different runs never overlap. Plan: start
the MPS daemon so new runs are MPS clients (the running ones finish unchanged), then re-test 3-4 runs per GPU; in
parallel a 16384-env run (`abl_B_rotlin_ent5e-3_env16k_r6k`) tests bigger batches per run.

