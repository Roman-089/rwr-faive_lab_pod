import gymnasium as gym

from . import agents

gym.register(
    id="orca-v2-inhand-repose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_env_cfg:OrcaV2ReposeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgOrcaV2",
    },
)

# Subtask 2 (exteroception): proprio task + cube pose in the policy observation (OBS_DIM 81).
gym.register(
    id="orca-v2-inhand-repose-cube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_v2_repose_cube_env_cfg:OrcaV2ReposeCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgOrcaV2Cube",
    },
)
