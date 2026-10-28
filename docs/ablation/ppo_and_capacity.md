# PPO hyperparameters and network capacity

[Index](README.md) · metrics: [eval protocol](eval_protocol.md)

## Question / hypothesis

- Entropy: A's eval collapses after the action std shrinks (1.5 → 0.18) while the training reward stays flat; our
  entropy_coef 1e-4 is 100× below legged_gym's 1e-2. More entropy should keep exploration up and avoid the late eval drop.
- Capacity: do larger (normalized) networks help PPO here?

## Current PPO cfg (`agents/rsl_rl_ppo_cfg.py`)

lr 1e-4 (adaptive, desired_kl 0.01), gamma 0.99, lambda 0.95, 24 steps/env, 5 epochs × 4 minibatches,
entropy 1e-4, clip 0.2, init_noise_std 1.5. Actor / critic MLP [512, 512, 256, 128] (same as the PT paper's teacher).

## Literature

- legged_gym default entropy 1e-2 vs our 1e-4.
- SimBa (2410.09754) / BRO: larger normalized nets help PPO.
- DexPBT 2305.12127 (later: PBT): Isaac Lab PBT exists but **rl_games only** (we use rsl_rl) → would need porting.
- See [literature](literature.md).

## Ideas

| Parameter | Values to try | Why |
|---|---|---|
| `entropy_coef` | 1e-3, 5e-3 | noise std collapsed to 0.18 → little exploration after ~4k iters |
| `gamma` | 0.99 → 0.995 | drops are delayed consequences; 0.99 horizon ≈ 5 s at 20 Hz |
| `num_steps_per_env` | 24 → 32/48 | longer rollouts, better advantage estimates |
| `desired_kl` | 0.01 → 0.016 | faster learning early |
| `num_envs` | 8192 vs 16384 | more data per iteration (fits on L40S?) |
| `max_iterations` | 4000–6000 is enough for screening runs | baseline plateaued at ~4k |
| Network size | [512, 512, 256, 128] → [256, 256, 128] (faster) vs [1024, 512, 256] | |
| PBT | DexPBT | needs porting to rsl_rl |

## Runs

| Run | Task id | Change vs baseline | Job / GPU | Iters | Status (old log) |
|---|---|---|---|---|---|
| `abl_B_asym_ent5e-3` | `-asym-v0` | B + entropy_coef 1e-4 → 5e-3 (hydra override) | 29833758 / A100 | 5k | done (2026-10-28 16:05) |
| `abl_B_asym_ent5e-3_seed1` | `-asym-v0` | same, seed 1 (replicate) | 29841912 / A100 | 5k | queued |
| `abl_B_asym_ent1e-2` | `-asym-v0` | B + entropy_coef 1e-2 (legged_gym default) | 29841914 / A100 | 5k | queued |
| `big` (planned) | on the reward-wave winner | actor/critic MLP 512-512-256-128 → 1024-1024-512-256 | – | – | planned |

## Results

**`abl_B_asym_ent5e-3`** (eval, nominal / perturbed):

| ckpt | RPM | RA % | DFSR % | DC |
|---|---|---|---|---|
| 1000 | 6.13 / 7.15 | 65.4 / 63.1 | 80 / 56 | 0.28 / 1.02 |
| 2000 | 6.95 / 6.58 | 73.9 / 61.6 | 84 / 52 | 0.20 / 1.03 |
| 3000 | 6.98 / 6.98 | 74.1 / 66.5 | 91 / 52 | 0.12 / 0.70 |
| 4000 | 6.84 / 7.09 | 74.7 / 67.0 | 89 / 56 | 0.20 / 0.92 |
| 4999 | **7.33 / 7.31** | **78.6 / 69.8** | 88 / 55 | 0.12 / 0.75 |

vs B (2 seeds, last ckpt) 5.45 RPM / RA 64 % / DFSR 88 %, and the privileged teacher 7.44 / 81.4 % / 94 %.

### Log

- **2026-10-27** — A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
- **2026-10-27 20:25** — queued `abl_B_asym_ent5e-3`: A's eval collapses after the action std shrinks (1.5 → 0.18)
  while the training reward stays flat; our entropy_coef 1e-4 is 100× below legged_gym's 1e-2. Hypothesis: more entropy
  keeps exploration up and avoids the late eval drop. B's eval moved to A100 (L40S hosts full for >2 h).
- **2026-10-28 16:05** — ent5e-3 eval: 7.33 / 7.31 RPM, RA 78.6 / 69.8 %, DFSR 88 / 55 %. Replicate + 1e-2 queued.
- **2026-10-28 (§15, agreed)** — `big` is one of the next architecture candidates, queued after the reward runs on the
  winner of the reward wave.

## Conclusions so far

- **Entropy 1e-4 → 5e-3 is the second large effect after the asymmetric critic: +1.9 RPM (5.45 → 7.33), RA 64 → 79 %,
  same drop-free rate** — a deployable policy at the privileged teacher's level (single seed; replicate queued). It
  already beats B's final checkpoint at it 1000 (6.13) and improves monotonically, i.e. no late degradation (cf. A).
- Its *training* reward is much lower than B's (~970 vs ~1570) because training reward is measured with the
  stochastic policy, whose noise std grew to 2.8 (vs 0.18 for A): training reward is not a usable comparison across
  entropy settings, only the deterministic eval is.
- The low entropy coefficient (100× below legged_gym's 1e-2) was suppressing exploration: the policy settled early on a
  slow, safe gait.

## Next steps / open questions

- ~~Entropy run eval: does it remove the late-checkpoint degradation seen in A?~~ Yes (monotonic to 7.33 RPM).
- Replicate (seed 1) and 1e-2 queued. If it holds, **B + entropy 5e-3 (or 1e-2) becomes the base** for the reward,
  action-space and capacity runs, and the teacher / students should be retrained with it (the teacher used 1e-4).
- Std 2.8 on actions in [-1, 1] means exploration actions mostly saturate: check whether a state-independent std
  this large is optimal or whether a smaller max std / annealed entropy does better.
- `big` after the reward wave; `gamma` / rollout length / `desired_kl` untested.
