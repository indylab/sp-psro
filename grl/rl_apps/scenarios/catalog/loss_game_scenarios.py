from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer, PPOTorchPolicy

from grl.algos.nfsp_rllib.nfsp import NFSPTrainer, NFSPTorchAveragePolicy
from grl.envs.loss_game_alpha_multi_agent_env import LossGameAlphaMultiAgentEnv
from grl.envs.loss_game_multi_agent_env import LossGameMultiAgentEnv
from grl.envs.multi_dim_loss_game_alpha_multi_agent_env import LossGameMultiDimMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.nfsp_scenario import NFSPScenario
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.stopping_conditions import NoStoppingCondition
from grl.rl_apps.scenarios.trainer_configs.loss_game_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched

for total_moves in [2, 5, 10]:
    for game_class in [LossGameMultiAgentEnv, LossGameAlphaMultiAgentEnv, LossGameMultiDimMultiAgentEnv]:
        # loss_game_psro_2_moves
        # loss_game_psro_10_moves
        # loss_game_psro_10_moves_alpha

        # loss_game_psro_10_moves_multi_dim

        game_variant = "_alpha" if issubclass(game_class, LossGameAlphaMultiAgentEnv) else ""
        game_variant = "_multi_dim" if issubclass(game_class, LossGameMultiDimMultiAgentEnv) else game_variant

        base_psro = PSROScenario(
            name=f"loss_game_psro_{total_moves}_moves{game_variant}",
            ray_cluster_cpus=default_if_creating_ray_head(default=11),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            env_class=game_class,
            env_config={
                "total_moves": total_moves
            },
            mix_metanash_with_uniform_dist_coeff=0.0,
            allow_stochastic_best_responses=True,
            trainer_class=PPOTrainer,
            policy_classes={
                "metanash": PPOTorchPolicy,
                "best_response": PPOTorchPolicy,
                "eval": PPOTorchPolicy,
            },
            num_eval_workers=5,
            games_per_payoff_eval=1000,
            p2sro=False,
            p2sro_payoff_table_exponential_avg_coeff=None,
            p2sro_sync_with_payoff_table_every_n_episodes=None,
            single_agent_symmetric_game=False,
            get_trainer_config=loss_game_psro_ppo_params,

            # NXDO params
            # psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
            #     br_policy_id="best_response",
            #     dont_check_plateau_before_n_episodes=int(2e4),
            #     check_plateau_every_n_episodes=int(2e4),
            #     minimum_reward_improvement_otherwise_plateaued=0.01,
            #     max_train_episodes=int(1e5),
            # ),

            # AO PSRO params (no early stopping)
            psro_get_stopping_condition=lambda: EpisodesSingleBRRewardPlateauStoppingCondition(
                br_policy_id="best_response",
                dont_check_plateau_before_n_episodes=int(1e5),
                check_plateau_every_n_episodes=int(1e5),
                minimum_reward_improvement_otherwise_plateaued=0.01,
                max_train_episodes=int(1e5),
            ),
            calc_exploitability_for_openspiel_env=False,
        )
        scenario_catalog.add(base_psro)

        # loss_game_nxdo_2_moves
        # loss_game_nxdo_10_moves
        # loss_game_nxdo_10_moves_alpha

        if issubclass(game_class, LossGameMultiDimMultiAgentEnv):
            # loss_game_psro_10_moves_multi_dim_max_move_3
            # loss_game_psro_10_moves_multi_dim_max_move_1.5
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_3",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 3.0,
                },
            ))
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_1.5",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 1.5,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_1.5_10_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_1.5_10_dim",
                env_config={
                    "dim": 10,
                    "total_moves": total_moves,
                    "max_move_amount": 1.5,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_2_10_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_2_10_dim",
                env_config={
                    "dim": 10,
                    "total_moves": total_moves,
                    "max_move_amount": 2.0,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_1_10_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_1_10_dim",
                env_config={
                    "dim": 10,
                    "total_moves": total_moves,
                    "max_move_amount": 1.0,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_0p5_10_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_0p5_10_dim",
                env_config={
                    "dim": 10,
                    "total_moves": total_moves,
                    "max_move_amount": 0.5,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_0p25_10_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_0p25_10_dim",
                env_config={
                    "dim": 10,
                    "total_moves": total_moves,
                    "max_move_amount": 0.25,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_0.1_16_dim
            # loss_game_psro_sac_10_moves_multi_dim_max_move_0.1_16_dim
            # loss_game_psro_sac_10_moves_multi_dim_max_move_0.1_16_dim_longer_training


            dim16_psro = base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_0.1_16_dim",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 0.1,
                    "dim": 16,
                },
            )
            scenario_catalog.add(dim16_psro)

            # loss_game_psro_10_moves_multi_dim_max_move_1_16_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_1_16_dim",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 1.0,
                    "dim": 16,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_2_16_dim
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_2_16_dim",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 2.0,
                    "dim": 16,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_2.5_16_dim_independent_sins
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_2.5_16_dim_independent_sins",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 2.5,
                    "use_independent_sins": True,
                    "dim": 16,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_3_16_dim_independent_sins
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_3_16_dim_independent_sins",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 3.0,
                    "use_independent_sins": True,
                    "dim": 16,
                },
            ))

            # loss_game_psro_10_moves_multi_dim_max_move_1.5_16_dim_independent_sins
            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_max_move_1.5_16_dim_independent_sins",
                env_config={
                    "total_moves": total_moves,
                    "max_move_amount": 1.5,
                    "use_independent_sins": True,
                    "dim": 16,
                },
            ))

        # loss_game_nfsp_2_moves
        # loss_game_nfsp_10_moves
        # loss_game_nfsp_10_moves_alpha

        base_nfsp = NFSPScenario(
            name=f"loss_game_nfsp_{total_moves}_moves{game_variant}",
            ray_cluster_cpus=default_if_creating_ray_head(default=4),
            ray_cluster_gpus=default_if_creating_ray_head(default=0),
            ray_object_store_memory_cap_gigabytes=1,
            env_class=game_class,
            env_config={
                "total_moves": total_moves,
                "discrete_actions_for_players": [0, 1],
            },
            trainer_class=DQNTrainer,
            avg_trainer_class=NFSPTrainer,
            policy_classes={
                "average_policy": NFSPTorchAveragePolicy,
                "best_response": SimpleQTorchPolicyPatched,
            },
            get_trainer_config=loss_game_nfsp_dqn_params,
            get_avg_trainer_config=loss_game_nfsp_avg_policy_params,
            anticipatory_param=0.1,
            nfsp_get_stopping_condition=lambda: NoStoppingCondition(),
            calculate_openspiel_metanash=False,
            calculate_openspiel_metanash_at_end=False,
            calc_metanash_every_n_iters=0,
            checkpoint_every_n_iters=500
        )
        scenario_catalog.add(base_nfsp)

        if issubclass(game_class, LossGameMultiDimMultiAgentEnv):
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_3",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 3.0,
                },
            ))
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_1.5",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 1.5,
                },
            ))
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_1.5_discrete_6",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 1.5,
                    "num_actions_per_dim": 6,
                },
            ))

            # loss_game_nfsp_10_moves_multi_dim_max_move_1.5_dim_10
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_1.5_dim_10",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 1.5,
                    "num_actions_per_dim": 2,
                    "dim": 10,
                },
            ))

            # loss_game_nfsp_10_moves_multi_dim_max_move_0.5_dim_10
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_0.5_dim_10",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 0.5,
                    "num_actions_per_dim": 2,
                    "dim": 10,
                },
            ))

            # loss_game_nfsp_10_moves_multi_dim_max_move_0.25_dim_10
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_max_move_0.25_dim_10",
                env_config={
                    "total_moves": total_moves,
                    "discrete_actions_for_players": [0, 1],
                    "max_move_amount": 0.25,
                    "num_actions_per_dim": 2,
                    "dim": 10,
                },
            ))

        for alpha in [2.5, 2.7, 2.9]:

            # loss_game_psro_10_moves_alpha_2.5
            # loss_game_psro_10_moves_alpha_2.7
            # loss_game_psro_10_moves_alpha_2.9

            # loss_game_nfsp_10_moves_alpha_2.5
            # loss_game_nfsp_10_moves_alpha_2.7
            # loss_game_nfsp_10_moves_alpha_2.9

            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_psro_{total_moves}_moves{game_variant}_{alpha:.1f}",
                env_config={
                    "total_moves": total_moves,
                    "alpha": alpha,
                },
            ))

            scenario_catalog.add(base_psro.with_updates(
                name=f"loss_game_rn_psro_{total_moves}_moves{game_variant}_{alpha:.1f}",
                env_config={
                    "total_moves": total_moves,
                    "alpha": alpha,
                },
                ray_cluster_cpus=default_if_creating_ray_head(default=42),
            ))

            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_{alpha:.1f}",
                env_config={
                    "total_moves": total_moves,
                    "alpha": alpha,
                    "discrete_actions_for_players": [0, 1],
                },
            ))

            # loss_game_nfsp_10_moves_alpha_2.7_10_actions_per_dim
            scenario_catalog.add(base_nfsp.with_updates(
                name=f"loss_game_nfsp_{total_moves}_moves{game_variant}_{alpha:.1f}_10_actions_per_dim",
                env_config={
                    "total_moves": total_moves,
                    "alpha": alpha,
                    "discrete_actions_for_players": [0, 1],
                    "num_actions_per_dim": 10,
                },
            ))
