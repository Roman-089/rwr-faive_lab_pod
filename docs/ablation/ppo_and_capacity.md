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
| `abl_B_asym_ent5e-3` | `-asym-v0` | B + entropy_coef 1e-4 → 5e-3 (hydra override) | 29833758 / A100 | 5k | running |
| `big` (planned) | on the reward-wave winner | actor/critic MLP 512-512-256-128 → 1024-1024-512-256 | – | – | planned |

## Results

None yet.

### Log

- **2026-10-27** — A plateaus at reward ~490–510 from iter ~4k; action noise std collapses to 0.18.
- **2026-10-27 20:25** — queued `abl_B_asym_ent5e-3`: A's eval collapses after the action std shrinks (1.5 → 0.18)
  while the training reward stays flat; our entropy_coef 1e-4 is 100× below legged_gym's 1e-2. Hypothesis: more entropy
  keeps exploration up and avoids the late eval drop. B's eval moved to A100 (L40S hosts full for >2 h).
- **2026-10-28 (§15, agreed)** — `big` is one of the next architecture candidates, queued after the reward runs on the
  winner of the reward wave.

## Conclusions so far

- None yet.

## Next steps / open questions

- Entropy run eval: does it remove the late-checkpoint degradation seen in A?
- `big` after the reward wave; `gamma` / rollout length / `desired_kl` untested.
