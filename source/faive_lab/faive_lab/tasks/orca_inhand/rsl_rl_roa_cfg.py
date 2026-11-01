"""rsl_rl runner cfg for ROA (roa.py, docs/ablation/roa.md): PPORunnerCfgOrcaV2's PPO hyperparameters and network sizes,
with the reference ROA modules and terms (~/go2-flyball-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py
l. 621-650: priv_encoder_dims [64, 20], tanh_encoder_output False, dagger_update_freq 20, priv_reg_coef_schedual).

Schedule: the reference uses [0, 0.1, 5000, 3000] (coefficient 0 -> 0.1 between counter 5000 and 8000). Our runs are
~6000 iterations, so the ramp starts at counter 2500 and ends at 4500. The counter advances by 1 per update and by 1
more per DAgger update (every 20th iteration), i.e. ~1.05 per iteration: the ramp covers iterations ~2380-4290.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .agents.rsl_rl_ppo_cfg import PPORunnerCfgOrcaV2


@configclass
class RslRlPpoActorCriticROACfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticROA"
    history_length: int = 20
    priv_encoder_dims: list = [64, 20]
    tanh_encoder_output: bool = False


@configclass
class RslRlPpoROAAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "PPOROA"
    dagger_update_freq: int = 20
    priv_reg_coef_schedual: list = [0.0, 0.1, 2500, 2000]


@configclass
class PPORunnerCfgOrcaV2ROA(PPORunnerCfgOrcaV2):
    """ROA on the -servofast recipe: actor = backbone([proprio_t, latent]); critic = B's privileged group + priv."""

    obs_groups = {"policy": ["policy"], "critic": ["critic", "priv"], "priv": ["priv"]}
    policy = RslRlPpoActorCriticROACfg(
        init_noise_std=1.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 512, 256, 128],
        critic_hidden_dims=[512, 512, 256, 128],
        activation="elu",
        history_length=20,
        priv_encoder_dims=[64, 20],
        tanh_encoder_output=False,
    )
    algorithm = RslRlPpoROAAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0.0, 0.1, 2500, 2000],
    )
