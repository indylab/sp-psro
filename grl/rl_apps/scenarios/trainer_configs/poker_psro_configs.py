import os
from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.models.torch.torch_action_dist import TorchBeta
from ray.rllib.models.torch.torch_action_dist import TorchSquashedGaussian
from ray.rllib.utils import merge_dicts
from ray.tune.registry import RLLIB_ACTION_DIST, _global_registry

from grl.rl_apps.scenarios.trainer_configs.defaults import GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, \
    GRL_DEFAULT_POKER_PPO_PARAMS
from grl.rllib_tools.action_dists import TorchGaussianSquashedGaussian
from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy

class _PokerAndOshiBetaTorchDist(TorchBeta):
    def __init__(self, inputs, model):
        super(_PokerAndOshiBetaTorchDist, self).__init__(inputs, model, low=-1.0, high=1.0)


_global_registry.register(RLLIB_ACTION_DIST, "PokerAndOshiBetaTorchDist", _PokerAndOshiBetaTorchDist)
_global_registry.register(RLLIB_ACTION_DIST, "TorchGaussianSquashedGaussian", TorchGaussianSquashedGaussian)
_global_registry.register(RLLIB_ACTION_DIST, "TorchSquashedGaussian", TorchSquashedGaussian)


def psro_kuhn_dqn_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS


def psro_leduc_dqn_params_openspiel(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.06,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(20e6) * 10,  # Timesteps over which to anneal epsilon.
            # "epsilon_timesteps": int(20e6) * 1000000000,  # Timesteps over which to anneal epsilon. #TODO change back

        },

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 19200 * 10,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    })


def psro_leduc_ppo_params_old(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 20000,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": None,
        }),

    })

def psro_leduc_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
          "batch_mode": "complete_episodes",
          "clip_param": 0.3520894737412451,
          "entropy_coeff": 0.001,
          "framework": "torch",
          "gamma": 1.0,
          "grad_clip": 40.0,
          "kl_coeff": 0.0035728559740701616,
          "kl_target": 0.009277655373078572,
          "lambda": 0.9,
          "lr": 0.007453255776317367,
          "lr_schedule": None,
          "metrics_smoothing_episodes": 5000,
            "model": merge_dicts(MODEL_DEFAULTS, {
                "custom_action_dist": "PokerAndOshiBetaTorchDist",
                "fcnet_activation": "relu",
                "fcnet_hiddens": [128, 128],
                "custom_model": None,
            }),
          "num_envs_per_worker": 1,
          "num_gpus": 0.0,
          "num_gpus_per_worker": 0.0,
          "num_sgd_iter": 30,
          "num_workers": 4,
          "observation_filter": "NoFilter",
          "rollout_fragment_length": 400,
          "sgd_minibatch_size": 128,
          "train_batch_size": 1024,
          "use_critic": True,
          "use_gae": True,
          "vf_clip_param": 1000,
          "vf_share_layers": False
        })


def psro_leduc_ppo_discrete_action_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "clip_param": 0.03950352730537203,
        "entropy_coeff": 0.001,
        "framework": "torch",
        "grad_clip": 100.0,
        "kl_coeff": 0.004850738962363403,
        "kl_target": 0.0028703697479350184,
        "lambda": 1.0,
        "lr": 6.722186143685379e-05,
        "metrics_smoothing_episodes": 5000,
        "num_envs_per_worker": 1,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0,
        "num_sgd_iter": 10,
        "num_workers": 4,
        "observation_filter": "NoFilter",
        "rollout_fragment_length": 1,
        "sgd_minibatch_size": 128,
        "timesteps_per_iteration": 0,
        "train_batch_size": 2048,
        "vf_clip_param": 1000,
        "vf_loss_coeff": 0.01,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        },
    }


def psro_oshi_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_POKER_PPO_PARAMS, {
        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,
        "metrics_smoothing_episodes": 5000,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [64, 64],
            "custom_model": None,
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),

    })


def larger_psro_oshi_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(psro_oshi_ppo_params(env=env), {
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": None,
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": [(0, 0.01), (int(2000e3), 0.0)],
    })


def larger_psro_oshi_ppo_params_lower_entropy(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(psro_oshi_ppo_params(env=env), {
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": None,
            "custom_action_dist": "TorchGaussianSquashedGaussian",
        }),
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": [(0, 0.00000), (int(200e3), 0.0)],
    })


def psro_leduc_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.
        },

        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 10000,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,

        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    })

def psro_leduc_dqn_params_constant_epsilon_0p2(env: MultiAgentEnv) -> Dict[str, Any]:
    params = psro_leduc_dqn_params(env)
    params["exploration_config"] = {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.2,
            "final_epsilon": 0.2,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.
        }
    return params

def psro_leduc_dqn_params_longer_2x_explore_anneal(env: MultiAgentEnv) -> Dict[str, Any]:
    params = psro_leduc_dqn_params(env)
    params["exploration_config"]["epsilon_timesteps"] = int(1e5) * 2
    return params

def psro_repeat_kuhn_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    # kuhn_repeat_10_hyper_param_search_dqn/DQN_RepeatPokerMultiAgentEnv_b8431bfa_25_adam_epsilon=0.01,batch_mode=truncate_episodes,buffer_size=50000,callbacks=<class '__main_2022-03-07_13-58-42
    return {
        "adam_epsilon": 0.01,
        "batch_mode": "truncate_episodes",
        "buffer_size": 50000,
        "compress_observations": True,
        "double_q": True,
        "dueling": False,

        "evaluation_config": {
            "explore": False
        },
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(2e5),  # Timesteps over which to anneal epsilon.
        },
        "explore": True,
        "final_prioritized_replay_beta": 0.0,
        "framework": "torch",
        "gamma": 1.0,
        "grad_clip": None,
        "hiddens": [
            256
        ],
        "learning_starts": 16000,
        "lr": 0.002349188141753868,
        "lr_schedule": None,
        "metrics_smoothing_episodes": 5000,
        "min_iter_time_s": 0,
        "n_step": 1,
        "noisy": False,
        "num_atoms": 1,

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,


        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.0,
        "prioritized_replay_beta": 0.0,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "prioritized_replay_eps": 0.0,
        "rollout_fragment_length": 32 * 32,
        "sigma0": 0.5,
        "target_network_update_freq": 1000,
        "timesteps_per_iteration": 0,
        "train_batch_size": 1024,
        "training_intensity": None,
        "v_max": 10.0,
        "v_min": -10.0,
        "worker_side_prioritization": False,
        "model": merge_dicts(MODEL_DEFAULTS, {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    }


def psro_kuhn_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(GRL_DEFAULT_OPENSPIEL_POKER_DQN_PARAMS, {
        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": ValidActionsEpsilonGreedy,
            # Config for the Exploration class constructor:
            "initial_epsilon": 0.20,
            "final_epsilon": 0.001,
            "epsilon_timesteps": int(1e5),  # Timesteps over which to anneal epsilon.
        },

        "num_gpus": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_workers": 4,
        "num_gpus_per_worker": float(os.getenv("WORKER_GPU_NUM", 0.0)),
        "num_envs_per_worker": 1,

        # How many steps of the model to sample before learning starts.
        "learning_starts": 16000,
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.q
        "rollout_fragment_length": 8 * 32,

        # Size of a batch sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 4096,
    })

def psro_kuhn_dqn_params_valid_actions(env: MultiAgentEnv) -> Dict[str, Any]:
    return merge_dicts(psro_kuhn_dqn_params(env=env), {
        "model": merge_dicts(MODEL_DEFAULTS, {
                "fcnet_activation": "relu",
                "fcnet_hiddens": [128],
                "custom_model": get_valid_action_fcn_class_for_env(env=env)
        }),
    })