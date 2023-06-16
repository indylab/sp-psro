from typing import Dict, Any

from ray.rllib.env import MultiAgentEnv

from grl.rllib_tools.models.valid_actions_fcnet import get_valid_action_fcn_class_for_env
from grl.rllib_tools.valid_actions_epsilon_greedy import ValidActionsEpsilonGreedy


def psro_goofspiel_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
            "adam_epsilon": 1e-08,
            "batch_mode": "truncate_episodes",
            "buffer_size": 200000,
            "compress_observations": True,
            "double_q": True,
            "dueling": False,

            "exploration_config": {
                "epsilon_timesteps": 2000000,
                "final_epsilon": 0.001,
                "initial_epsilon": 0.06,
                "type": ValidActionsEpsilonGreedy
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
            "lr": 0.0018513348642245236,
            "lr_schedule": None,
            "metrics_smoothing_episodes": 5000,
            # "min_iter_time_s": 2,
            "n_step": 1,
            "noisy": False,
            "num_atoms": 1,
            "num_envs_per_worker": 1,
            "num_workers": 4,
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.0,
            "prioritized_replay_beta": 0.0,
            "prioritized_replay_beta_annealing_timesteps": 20000,
            "prioritized_replay_eps": 0.0,
            "rollout_fragment_length": 256,
            "sigma0": 0.5,
            "target_network_update_freq": 100000,
            "timesteps_per_iteration": 0,
            "train_batch_size": 2048,
            "training_intensity": None,
            "v_max": 10.0,
            "v_min": -10.0,
            "worker_side_prioritization": False,

            "model": {
                "_time_major": False,
                "conv_activation": "relu",
                "conv_filters": None,
                "custom_action_dist": None,
                "custom_model": get_valid_action_fcn_class_for_env(env=env),
                "custom_model_config": {},
                "custom_preprocessor": None,
                "dim": 84,
                "fcnet_activation": "relu",
                "fcnet_hiddens": [
                    128,
                    128,
                    128
                ],
                "framestack": True,
                "free_log_std": False,
                "grayscale": False,
                "lstm_cell_size": 256,
                "lstm_use_prev_action_reward": False,
                "max_seq_len": 20,
                "no_final_linear": False,
                "use_lstm": False,
                "vf_share_layers": True,
                "zero_mean": True
            }

    }


def psro_goofspiel_ppo_discrete_action_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "clip_param": 0.010510880764393143,
        "entropy_coeff": 0.001,
        "framework": "torch",
        "grad_clip": 100.0,
        "kl_coeff": 0.011326815989171434,
        "kl_target": 0.01193937497368384,
        "lambda": 0.9,
        "lr": 0.0002583159872861569,
        "metrics_smoothing_episodes": 5000,
        "num_envs_per_worker": 1,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 10,
        "num_workers": 4,
        "observation_filter": "NoFilter",
        "rollout_fragment_length": 400,
        "sgd_minibatch_size": 128,
        "timesteps_per_iteration": 0,
        "train_batch_size": 512,
        "vf_clip_param": 1000,
        "vf_loss_coeff": 0.01,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }
    }


def psro_liars_dice_ppo_discrete_action_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        "clip_param": 0.024285372209276124,
        "entropy_coeff": 0.0001,
        "framework": "torch",
        "grad_clip": 100.0,
        "kl_coeff": 0.007282867994388229,
        "kl_target": 0.07466749827931334,
        "lambda": 1.0,
        "lr": 0.004538592900280396,
        "metrics_smoothing_episodes": 5000,
        "num_envs_per_worker": 1,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0,
        "num_sgd_iter": 5,
        "num_workers": 4,
        "observation_filter": "NoFilter",
        "rollout_fragment_length": 10,
        "sgd_minibatch_size": 64,
        "timesteps_per_iteration": 0,
        "train_batch_size": 1024,
        "vf_clip_param": 1000,
        "vf_loss_coeff": 0.01,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }
    }

def psro_goofspiel_dqn_params_longer(env: MultiAgentEnv) -> Dict[str, Any]:
    params = psro_goofspiel_dqn_params(env).copy()
    params["exploration_config"] = {
                "epsilon_timesteps": 3000000,
                "final_epsilon": 0.001,
                "initial_epsilon": 0.06,
                "type": ValidActionsEpsilonGreedy
            }
    return params

def psro_liars_dice_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    return {
        
            "adam_epsilon": 1e-06,
            "batch_mode": "truncate_episodes",
            "buffer_size": 50000,
            "compress_observations": True,
            "double_q": True,
            "dueling": False,
            "exploration_config": {
                "epsilon_timesteps": 200000,
                "final_epsilon": 0.001,
                "initial_epsilon": 0.06,
                "type": ValidActionsEpsilonGreedy
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
            "lr": 0.002568159322888987,
            "lr_schedule": None,
            "metrics_smoothing_episodes": 5000,
            "min_iter_time_s": 2,
            "n_step": 1,
            "noisy": False,
            "num_atoms": 1,
            "num_envs_per_worker": 1,
            "num_workers": 4,
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.0,
            "prioritized_replay_beta": 0.0,
            "prioritized_replay_beta_annealing_timesteps": 20000,
            "prioritized_replay_eps": 0.0,
            "rollout_fragment_length": 512,
            "sigma0": 0.5,
            "target_network_update_freq": 100,
            "timesteps_per_iteration": 0,
            "train_batch_size": 4096,
            "training_intensity": None,
            "v_max": 10.0,
            "v_min": -10.0,
            "worker_side_prioritization": False,

            "model": {
                "_time_major": False,
                "conv_activation": "relu",
                "conv_filters": None,
                "custom_action_dist": None,
                "custom_model": get_valid_action_fcn_class_for_env(env=env),
                "custom_model_config": {},
                "custom_preprocessor": None,
                "dim": 84,
                "fcnet_activation": "relu",
                "fcnet_hiddens": [
                    128,
                    128
                ],
                "framestack": True,
                "free_log_std": False,
                "grayscale": False,
                "lstm_cell_size": 256,
                "lstm_use_prev_action_reward": False,
                "max_seq_len": 20,
                "no_final_linear": False,
                "use_lstm": False,
                "vf_share_layers": True,
                "zero_mean": True
            }
    }


def psro_battleship_dqn_params(env: MultiAgentEnv) -> Dict[str, Any]:
    # /home/jblanier/git/grl/grl/data/battleship_hyper_param_search_dqn_03.31.22PM_Aug-28-2022/DQN_BattleshipMultiAgentEnv_c46bcc4e_80_adam_epsilon=1e-06,batch_mode=truncate_episodes,buffer_size=50000,callbacks=<class '__main_2022-08-29_04-12-17
    return {
        "adam_epsilon": 1e-06,
        "batch_mode": "truncate_episodes",
        "buffer_size": 50000,
        "compress_observations": True,
        "double_q": True,
        "dueling": False,
        "evaluation_config": {
            "explore": False
        },
        "exploration_config": {
            "epsilon_timesteps": 10000.0,
            "final_epsilon": 0.0015737205285266658,
            "initial_epsilon": 1.0,
            "type": ValidActionsEpsilonGreedy
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
        "lr": 0.0054176501384288304,
        "lr_schedule": None,
        "metrics_smoothing_episodes": 5000,
        "n_step": 1,
        "noisy": False,
        "num_atoms": 1,
        "num_envs_per_worker": 1,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_workers": 4,
        "prioritized_replay": False,
        "prioritized_replay_alpha": 0.0,
        "prioritized_replay_beta": 0.0,
        "prioritized_replay_beta_annealing_timesteps": 20000,
        "prioritized_replay_eps": 0.0,
        "rollout_fragment_length": 8,
        "sigma0": 0.5,
        "target_network_update_freq": 1000,
        "timesteps_per_iteration": 0,
        "train_batch_size": 4096,
        "training_intensity": None,
        "v_max": 10.0,
        "v_min": -10.0,
        "worker_side_prioritization": False,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }
    }


def psro_battleship_ppo_params(env: MultiAgentEnv) -> Dict[str, Any]:
    # /home/jb/git/grl/grl/data/battleship_hyper_param_search_ppo_03.32.26PM_Aug-28-2022/PPO_BattleshipMultiAgentEnv_98c2db26_313_callbacks=<class '__main__.HyperParamSearchCallbacks'>,clip_param=0.016279,entropy_coeff=_2022-08-29_07-56-29
    return {
        "clip_param": 0.016278838885264126,
        "entropy_coeff": 0.01,
        "framework": "torch",
        "gamma": 1.0,
        "grad_clip": 10.0,
        "kl_coeff": 0.003415874212896736,
        "kl_target": 0.023591466652166538,
        "lambda": 1.0,
        "lr": 0.00020482127502261262,
        "metrics_smoothing_episodes": 5000,
        "num_envs_per_worker": 1,
        "num_gpus": 0.0,
        "num_gpus_per_worker": 0.0,
        "num_sgd_iter": 5,
        "num_workers": 4,
        "observation_filter": "NoFilter",
        "rollout_fragment_length": 10,
        "sgd_minibatch_size": 32,
        "timesteps_per_iteration": 0,
        "train_batch_size": 128,
        "vf_clip_param": 100,
        "vf_loss_coeff": 1.0,
        "model": {
            "fcnet_activation": "relu",
            "fcnet_hiddens": [128, 128, 128],
            "custom_model": get_valid_action_fcn_class_for_env(env=env),
        }
    }
