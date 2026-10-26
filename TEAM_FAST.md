# Team FAST – who edits what in this fork

Full table: `OWNERSHIP.md` in [rwr-fast/rwr-fast](https://github.com/rwr-fast/rwr-fast/blob/main/OWNERSHIP.md).

| File | Owner | Task ID |
|---|---|---|
| `source/faive_lab/faive_lab/assets/orca_v2.py` (joint params) | S1 – SysID | – |
| `tasks/orca_inhand/orca_v2_repose_env_cfg.py` | S0 – sim + env tuning (Subtask 1) | `orca-v2-inhand-repose-v0` → logs `orca_v2_repose/` |
| `tasks/orca_inhand/orca_v2_repose_cube_env_cfg.py` | E – exteroception (Subtask 2), inherits S0's config | `orca-v2-inhand-repose-cube-v0` → logs `orca_v2_repose_cube/` |
| `tasks/orca_inhand/mdp/`, `agents/` | S0 (E only adds new files/classes) | – |
| `scripts/motor_param_characterization/` | S1 | – |

Train the cube task: `python scripts/rsl_rl/train.py --task=orca-v2-inhand-repose-cube-v0 --headless --num_envs 8192 --logger wandb`
PRs: target **rwr-fast/rwr_faive_lab**, not srl-ethz. Course updates: `git fetch upstream && git merge upstream/main`.
