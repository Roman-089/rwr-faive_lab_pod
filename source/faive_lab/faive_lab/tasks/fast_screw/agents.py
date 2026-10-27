# Owner: X (Task X / FAST)
from isaaclab.utils import configclass

from faive_lab.tasks.orca_inhand.agents.rsl_rl_ppo_cfg import PPORunnerCfgOrcaV2


@configclass
class PPORunnerCfgFastScrew(PPORunnerCfgOrcaV2):
    """Same PPO settings as Task 1, own log folder logs/rsl_rl/fast_screw/."""

    experiment_name = "fast_screw"
    max_iterations = 5000
