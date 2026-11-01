# ROA: Regularized Online Adaptation (2026-11-01)

One-stage adaptation module: a privileged encoder and a history encoder trained together, with a regulariser that
keeps the privileged latent inferable from the observation history. Ported from the user's extreme-parkour-style
rsl_rl in `~/go2-flyball-parkour` (Cheng et al., *Extreme Parkour with Legged Robots*, 2023; ROA from Fu, Cheng, Pathak,
*Deep Whole-Body Control*, CoRL 2022) onto our rsl_rl 3.1.2 / Isaac Lab setup. **Code and CPU checks only; not run on
GPU yet.**

## Reference and what was mirrored

| reference (`~/go2-flyball-parkour`) | ours |
|---|---|
| `rsl_rl/rsl_rl/modules/actor_critic.py` l. 41-86 `StateHistoryEncoder` (Linear → 30 ch, two 1D convs for 20 steps, Linear → latent, ELU everywhere) | `roa.StateHistoryEncoder`, copied unchanged |
| same file l. 110-121 `priv_encoder` (Linear + activation per layer, the last included), `priv_encoder_dims` [64, 20] | `roa.mlp_with_final_activation`, [64, 20] |
| same file l. 142-157 actor backbone on [proprio, scan, priv_explicit, latent]; l. 197-203 `infer_priv_latent` / `infer_hist_latent` | `ActorCriticROA.backbone` on [proprio_t, latent] (no scan, no priv_explicit), `infer_priv_latent` / `infer_hist_latent` |
| `rsl_rl/rsl_rl/algorithms/ppo.py` l. 229-235, 288-291: `priv_reg_loss = ‖z_priv − sg(z_hist)‖₂` (L2 norm, mean over the batch) in the PPO loss with `priv_reg_coef` from `priv_reg_coef_schedual = [start, end, start_iter, ramp_iters]` | `PPOROA.update` (rsl_rl 3.1.2's PPO update without RND / symmetry, which we do not use) |
| same file l. 322-348 `update_dagger`: history encoder only, `‖sg(z_priv) − z_hist‖₂`, own Adam at the PPO learning rate; l. 400-401 `update_counter` in both | `PPOROA.update_dagger`, same counter semantics |
| `rsl_rl/rsl_rl/runners/on_policy_runner.py` l. 209, 219, 259-261: `hist_encoding = it % dagger_update_freq == 0`; that rollout *acts* with the history latent, then `update()` and `update_dagger()` | `PPOROA` keeps the rollout counter and sets `policy.hist_encoding` for the next rollout (the 3.1.2 runner has no such hook) |
| `legged_gym/legged_gym/envs/base/legged_robot_config.py` l. 621-650: `priv_encoder_dims [64, 20]`, `tanh_encoder_output False`, `dagger_update_freq 20`, `priv_reg_coef_schedual [0, 0.1, 5000, 3000]` | `rsl_rl_roa_cfg.py`: same, except the schedule below |

Kept from our pipeline:
- rsl_rl's empirical normaliser on the 74-dim per-step proprio. The reference scales observations in the env.
- rsl_rl's Gaussian with log-std (init 1.5). The reference uses a plain std.
- The critic: B's privileged group (81) plus `priv`.
- All PPO hyperparameters of `PPORunnerCfgOrcaV2`.

Reference quirk kept: on DAgger iterations the rollout acts with the history latent, but `update()` recomputes the
log-probs with the privileged latent, so the PPO ratio of those 1-in-20 iterations is slightly off-policy.

**Schedule:** `[0, 0.1, 2500, 2000]`. The coefficient is 0 until counter 2500, then rises linearly to 0.1 at 4500. The
counter advances by 1 per update and by 1 more per DAgger update, so about 1.05 per iteration. The ramp therefore spans
iterations ≈ 2380–4290 of a ~6000-iteration run. The reference starts at 5000 of its longer runs.

## Shapes

| tensor | size |
|---|---|
| `policy` obs (actor and history encoder) | 20 × 74 = 1480 (Isaac Lab term-major flat history, current step last) |
| proprio_t (backbone input) | 74 (last history step, normalised) |
| `priv` obs | 13 (`roa_obs.roa_dr_params`) |
| latent z_priv / z_hist | 20 |
| backbone | 94 → 512-512-256-128 ELU → 17 |
| critic | 81 + 13 = 94 → 512-512-256-128 ELU → 1 |
| history encoder | ~3 k parameters |

### Privileged vector (`roa_obs.roa_dr_params`, 13 values, each normalised to [-1, 1] from its cfg range)

| idx | parameter | source |
|---|---|---|
| 0-2 | cube static friction, dynamic friction, restitution | `object_physics_material` (reset), via the existing `domain_randomization_params` term |
| 3 | cube mass scale | `object_scale_mass` (reset), same |
| 4 | mean finger / skin friction (non-palm shapes) | `robot_physics_material` (startup), read from PhysX |
| 5 | mean palm / PLA link friction | `palm_physics_material` (+ `pla_physics_material` if present), shape ids from the event term |
| 6-7 | mean joint stiffness and damping scale | `robot_joint_stiffness_and_damping` (reset), via `domain_randomization_params` |
| 8-10 | servo model: delay substeps, mean acceleration-limit scale, mean play scale | `ServoProfileAction` per-env buffers (`adr.py`) |
| 11-12 | mount roll, pitch | `reset_robot_base` (root orientation vs default; pitch 0 unless randomised) |

Not included:
- **The 17 joint-encoder offsets.** They are an unobservable calibration error the history cannot recover, so ROA's
  regulariser would have to suppress them again.
- **The cube pose and velocity.** They are state, not parameters. A later variant could add a cube-state estimator
  (extreme-parkour's `priv_explicit`).

## Files

- `source/faive_lab/faive_lab/tasks/orca_inhand/roa.py`: `StateHistoryEncoder`, `ActorCriticROA`, `PPOROA`, `register()`.
- `source/faive_lab/faive_lab/tasks/orca_inhand/roa_obs.py`: the `priv` term.
- `source/faive_lab/faive_lab/tasks/orca_inhand/rsl_rl_roa_cfg.py`: runner cfg.
- `orca_v2_repose_ablation_env_cfg.py`: `ROAPrivObsGroupCfg`, `...ServoFastROAEnvCfg` (policy history 20 + `priv`).
- `__init__.py`: task `orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-roa-v0`.
- `scripts/export/export_history_policy.py`: `ROABody` (deploy path = history encoder + backbone), so
  `--run <roa run>` writes `policy_history.pt`. It needs the history-capable node with the `reset()` line.

## CPU checks (2026-11-01, stubbed Isaac, scratchpad `chk_roa.py`, `chk_roa_obs.py`)

- **Configs:** the env cfg, runner cfg and gym registration resolve. Policy history is 20, `priv` = `roa_dr_params`, the
  action is a `ServoProfileAction` with amax (1.5, 3.0) and delay (0, 8).
- **Forward / backward:** `act` (both latents), `evaluate`, `act_inference` and both latents have the right shapes, and
  the history encoder, privileged encoder and backbone all receive gradients.
- **Toy PPOROA run:** priv = a function of the history, 64 envs × 24 steps, 60 updates.
  - DAgger iterations 0 / 20 / 40, counter 63 after 60 updates (as in the reference).
  - DAgger loss 0.69 → 0.30 → 0.27, priv_reg 0.76 → 0.25.
  - The coefficient schedule gives 0 / 0 / 0.05 / 0.1 at counters 0 / 2500 / 3500 / 4500.
- **Export:** the TorchScript `HistoryPolicy` (`ROABody`) matches eager `act_inference` step by step over 30 steps (max
  |diff| 0.0), and `reset()` works.
- **`roa_dr_params` on a mocked env:** 13 values, all in [-1, 1], palm shapes separated from finger shapes, roll
  normalised.

Untested until a GPU run:
- the real PhysX reads (material shapes per body, default mass)
- the observation-manager reset order with the servo action buffers
- memory: storage holds 8192 × 24 × (1480 + 94 + 13) floats ≈ 1.25 GB
- the runner's eval path through `act_inference`, i.e. the history latent

## First run (not launched)

From scratch on the full -servofast recipe (06 base reward + calm + servo DR), ~6000 iterations:

```bash
# DTU
hpc/submit_ablation.sh orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-roa-v0 \
    roa_servofast_scratch 6000 gpul40s 10:00 \
    "env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0"
# pod (8192 envs, save every 100)
python scripts/rsl_rl/train.py --task orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-servofast-roa-v0 \
    --headless --num_envs 8192 --max_iterations 6000 --run_name roa_servofast_scratch agent.save_interval=100 \
    env.rewards.object_position_tracking.weight=45.0 env.rewards.cube_tilt.weight=-800.0 env.rewards.wrist_vel.weight=-16.0
```

Evaluate it with `eval_policy.py` on the same task, plus the friction / servo overrides. `act_inference` uses the
history latent, i.e. the deployed behaviour.

Open risks:
- **The calm recipe from scratch was weak before.** The realfric run from scratch reached only 64–84 % drop-free and
  swung the wrist. Every good policy so far came from the fast 06 base followed by a calm fine-tune.
- **A two-stage alternative needs another task.** ROA on the 06-base task first, then the calm + servo fine-tune, would
  need an ROA variant of `-onedir-droppen` with the same obs groups (not added).
- **The 20-dim latent may not carry much.** The 13 parameters mostly act through contact, and the hist3 actor has
  already shown that history adds speed.

## Staged runs (2026-11-01 13:50Z)

Every good policy came from the 06 base plus fine-tunes, and the calm recipe from scratch stayed at 64–84 % drop-free.
Servo randomization during calming made a third of the envs stall (`servofast_from_base`). So ROA follows the shipped
path in three stages, each a resume of the previous one:

1. `orca-v2-inhand-repose-asym-onedir-droppen-roa-v0`: the 06 base task, `env.rewards.object_position_tracking.weight=45.0`,
   4000 it from scratch (regularizer ramp 2500 → 4500 as in `rsl_rl_roa_cfg`).
2. `orca-v2-inhand-repose-asym-onedir-droppen-upright-wristpen-realfric-wristl1-roa-v0`: the calm recipe (tracking 45,
   tilt −800, wrist −16), 1400 it. Full regularizer from the start: `agent.algorithm.priv_reg_coef_schedual=[0.1,0.1,0,1]`.
3. `...-wristl1-servofast-roa-v0`: servo hardening, 400 it, same overrides and schedule.

Seed 42 runs on the pod, seed 1 on DTU.
