# Owner: X (Task X / FAST)
"""FAST hand (QDD index/thumb) – screw a bolt in and out fast. Train like Task 1:

    python scripts/rsl_rl/train.py --task=fast-screw-v0 --headless --num_envs 4096 --logger wandb --video
    python scripts/rsl_rl/play.py  --task=fast-screw-play-v0
"""
import gymnasium as gym

_ENV = f"{__name__}.fast_screw_env_cfg"
_PPO = f"{__name__}.agents:PPORunnerCfgFastScrew"

for task_id, cfg in (("fast-screw-v0", "FastScrewEnvCfg"),
                     ("fast-screw-60hz-v0", "FastScrew60HzEnvCfg"),
                     ("fast-screw-play-v0", "FastScrewPlayEnvCfg")):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{_ENV}:{cfg}", "rsl_rl_cfg_entry_point": _PPO},
    )
