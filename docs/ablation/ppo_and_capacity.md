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
| `abl_B_asym_ent5e-3_seed1` | `-asym-v0` | same, seed 1 (replicate) | 29841912 / A100 | 5k | done (2026-10-29 06:50) |
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

- **2026-10-29 06:50 — the entropy replicate (seed 1) is much weaker than seed 0** (nominal / perturbed):

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 5.77 / 5.86 | 62.4 / 53.0 | 77 / 50 | 0.31 / 0.80 |
  | 2000 | 5.74 / 5.47 | 67.1 / 56.4 | 86 / 52 | 0.17 / 0.78 |
  | 3000 | 5.51 / 5.64 | 65.1 / 60.0 | 83 / 66 | 0.25 / 0.55 |
  | 4000 | 5.54 / 5.61 | 64.7 / 55.8 | 73 / 63 | 0.38 / 0.80 |
  | 4999 | 5.74 / 5.45 | 69.9 / 59.0 | 88 / 56 | 0.13 / 0.55 |

  Two-seed means at 5k: **ent 5e-3 6.54 RPM (7.33 / 5.74) vs B 5.45 (5.66 / 5.23)**: +1.1 RPM, RA +7 pp, same DFSR.
  The two entropy seeds differ by 1.6 RPM, more than the gain, so the effect is likely positive but not established.
  Seed 1 also lacks seed 0's steady improvement (flat at ~5.6 from 1000 on).

- **2026-10-29 09:15 — entropy 1e-2** (`abl_B_asym_ent1e-2`, seed 0), nominal / perturbed:

  | ckpt | RPM | RA % | DFSR % | DC |
  |---|---|---|---|---|
  | 1000 | 5.47 / 6.34 | 57.6 / 54.8 | 42 / 28 | 1.23 / 2.41 |
  | 2000 | 6.69 / 6.81 | 73.4 / 66.7 | 80 / 53 | 0.27 / 0.75 |
  | 3000 | 6.84 / 7.30 | 76.2 / 70.0 | 88 / 63 | 0.20 / 0.56 |
  | 4000 | **7.69** / 7.65 | 78.5 / 71.4 | 84 / 63 | 0.19 / 0.64 |
  | 4999 | 7.60 / 7.47 | 79.4 / 69.7 | 75 / 48 | 0.30 / 0.88 |

  Entropy sweep on B at 5k (nominal RPM / DFSR): 1e-4 5.45 / 88 % (2 seeds), 5e-3 6.54 / 88 % (2 seeds), 1e-2 7.60 /
  75 % (1 seed). Speed rises with entropy, and at 1e-2 drops rise too. Within seed noise (±0.7 RPM) for the 5e-3 vs
  1e-2 step; `abl_B_rotlin_ent1e-2` repeats the comparison on the rotlin reward.

- **2026-10-29 19:20 — rotlin + entropy 1e-2 does not learn** (killed at it 640): the action noise std keeps growing
  (1.8 → 3.0), episodes stay at ~200 steps with 95 % drops. On the old reward the same entropy was at 450 steps / 40 % drops
  by it 600. With rotlin, 1e-2 is too much exploration; 5e-3 works (rotlin_ent5e-3: 9.50 RPM).

## Conclusions so far

- **Entropy 1e-4 → 5e-3: +1.1 RPM on two seeds (5.45 → 6.54; seeds 7.33 / 5.74), RA +7 pp, same drop-free rate**;
  seed 0 alone looked like +1.9 RPM. Probably positive, not established; it stays in the queued rotlin runs (no seed was
  worse than B), and `rotlin_ent5e-3` vs `rotlin` (10.41) is the next data point. It
  already beats B's final checkpoint at it 1000 (6.13) and improves monotonically, i.e. no late degradation (cf. A).
- Its *training* reward is much lower than B's (~970 vs ~1570) because training reward is measured with the
  stochastic policy, whose noise std grew to 2.8 (vs 0.18 for A): training reward is not a usable comparison across
  entropy settings, only the deterministic eval is.
- The low entropy coefficient (100× below legged_gym's 1e-2) was suppressing exploration: the policy settled early on a
  slow, safe gait.

## Next steps / open questions

- ~~Entropy run eval: does it remove the late-checkpoint degradation seen in A?~~ Yes (monotonic to 7.33 RPM).
- Replicate done (+1.1 RPM mean, high seed spread); 1e-2 still running. Tentatively **entropy 5e-3 is in the base** for the reward,
  action-space and capacity runs, and the teacher / students should be retrained with it (the teacher used 1e-4).
- Std 2.8 on actions in [-1, 1] means exploration actions mostly saturate: check whether a state-independent std
  this large is optimal or whether a smaller max std / annealed entropy does better.
- `big` after the reward wave; `gamma` / rollout length / `desired_kl` untested.
