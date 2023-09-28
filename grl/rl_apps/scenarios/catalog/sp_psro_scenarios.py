from ray.rllib.agents.dqn import DQNTrainer

from grl.envs.poker_multi_agent_env import PokerMultiAgentEnv
from grl.envs.goofspiel_multi_agent_env import GoofspielMultiAgentEnv, ARMACGoofspielMultiAgentEnv
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.catalog.common import default_if_creating_ray_head
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.stopping_conditions import *
from grl.rl_apps.scenarios.trainer_configs.goofspeil_and_liars_dice_configs import *
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched, SimpleQTorchPolicyPatchedSoftMaxSampling
from grl.rllib_tools.modified_policies.mixed_class_eval_policy import MixedClassEvalPolicyDQN, MixedClassEvalPolicyDQNSoftmax
from grl.envs.battleship_multi_agent_env import BattleshipMultiAgentEnv
from grl.envs.repeated_rps_multi_agent_env import RepeatedRPSMultiAgentEnv
from grl.envs.matrix_game_multi_agent_env import MatrixGameMultiAgentEnv

lairs_dice = PSROScenario(
    name="liars_dice_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=PokerMultiAgentEnv,
    env_config={
        "version": "liars_dice",
        'fixed_players': True,
        'dummy_action_multiplier': 1,
        'continuous_action_space': False,
        'penalty_for_invalid_actions': False,
        'append_valid_actions_mask_to_obs': True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_liars_dice_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6),
        check_plateau_every_n_steps=int(1.5e6),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(lairs_dice)


goofspiel = PSROScenario(
    name="goofspiel_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=GoofspielMultiAgentEnv,
    env_config={
        'version': "goofspiel",
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': True,
        'continuous_action_space': False,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6),
        check_plateau_every_n_steps=int(1.5e6),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6),
    ),
    calc_exploitability_for_openspiel_env=False,
)
scenario_catalog.add(goofspiel)

scenario_catalog.add(goofspiel.with_updates(
    name="goofspiel_rn_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=30),
))

scenario_catalog.add(goofspiel.with_updates(
    name="goofspiel_psro_dqn_longer",
    get_trainer_config=psro_goofspiel_dqn_params_longer,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(2.5e6),
        check_plateau_every_n_steps=int(2.5e6),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(2.5e6),
    ),
))


small_goofspiel = PSROScenario(
    name="small_goofspiel_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=ARMACGoofspielMultiAgentEnv,
    env_config={
        'version': "goofspiel",
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': True,
        'continuous_action_space': False,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6),
        check_plateau_every_n_steps=int(1.5e6),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(small_goofspiel)


scenario_catalog.add(small_goofspiel.with_updates(
    name="small_goofspiel_psro_dqn_gpu",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
))


small_goofspiel_shorter = small_goofspiel.with_updates(
    name="small_goofspiel_psro_dqn_shorter",
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6) // 2,
        check_plateau_every_n_steps=int(1.5e6) // 2,
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6) // 2,
    ),
)
scenario_catalog.add(small_goofspiel_shorter)


small_goofspiel_shorter_avg = small_goofspiel_shorter.with_updates(
    name="small_goofspiel_psro_dqn_shorter_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
)
scenario_catalog.add(small_goofspiel_shorter_avg)


lairs_dice_shorter = lairs_dice.with_updates(
    name="liars_dice_psro_dqn_shorter",
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6) // 2,
        check_plateau_every_n_steps=int(1.5e6) // 2,
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6) // 2,
    ),
)
scenario_catalog.add(lairs_dice_shorter)

scenario_catalog.add(lairs_dice_shorter.with_updates(
    name="liars_dice_psro_dqn_shorter_gpu",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
))

lairs_dice_shorter_avg = lairs_dice_shorter.with_updates(
    name="liars_dice_psro_dqn_shorter_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
)
scenario_catalog.add(lairs_dice_shorter_avg)

battleship = PSROScenario(
    name="battleship_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=BattleshipMultiAgentEnv,
    env_config={
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1.5e6)//2,
        check_plateau_every_n_steps=int(1.5e6)//2,
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1.5e6)//2,
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(battleship)

r_battleship = battleship.with_updates(
    name="r_battleship_psro_dqn",
    env_config={
        'version': "battleship",
        'fixed_players': True,
        'append_valid_actions_mask_to_obs': True,
        "allow_repeated_shots": True,
    },
    calc_exploitability_for_openspiel_env=False,
)
scenario_catalog.add(r_battleship)

scenario_catalog.add(battleship.with_updates(
    name="battleship_psro_dqn_gpu",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
))

scenario_catalog.add(battleship.with_updates(
    name="battleship_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))


scenario_catalog.add(r_battleship.with_updates(
    name="r_battleship_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))


scenario_catalog.add(r_battleship.with_updates(
    name="r_battleship_psro_dqn_softmax",
    policy_classes={
        "metanash": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "best_response": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "eval": SimpleQTorchPolicyPatchedSoftMaxSampling,
    },
))


scenario_catalog.add(r_battleship.with_updates(
    name="r_battleship_psro_dqn_softmax_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "best_response": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "eval": MixedClassEvalPolicyDQNSoftmax,
    },
))


scenario_catalog.add(battleship.with_updates(
    name="battleship_psro_dqn_goofspiel_params",
    get_trainer_config=psro_goofspiel_dqn_params,
))


repeated_rps = PSROScenario(
    name="repeated_rps_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=RepeatedRPSMultiAgentEnv,
    env_config={
        'fixed_players': True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(300e3),
        check_plateau_every_n_steps=int(300e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(300e3),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(repeated_rps)

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_biased_rps_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    env_config={
        'version': "spsro_biased_repeated_rps",
        'fixed_players': True,
    },
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_biased_rps_psro_dqn_softmax",
    env_config={
        'version': "spsro_biased_repeated_rps",
        'fixed_players': True,
    },
    policy_classes={
        "metanash": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "best_response": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "eval": SimpleQTorchPolicyPatchedSoftMaxSampling,
    },
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_biased_rps_psro_dqn",
    env_config={
        'version': "spsro_biased_repeated_rps",
        'fixed_players': True,
    },
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_biased_rps_psro_dqn_softmax_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    env_config={
        'version': "spsro_biased_repeated_rps",
        'fixed_players': True,
    },
    policy_classes={
        "metanash": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "best_response": SimpleQTorchPolicyPatchedSoftMaxSampling,
        "eval": MixedClassEvalPolicyDQNSoftmax,
    },
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_rps_psro_dqn_gpu",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_rps_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))


scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_rps_psro_dqn_avg_pol_better_params",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(600e3),
        check_plateau_every_n_steps=int(600e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(600e3),
    ),
))

scenario_catalog.add(repeated_rps.with_updates(
    name="repeated_rps_psro_dqn_avg_pol_better_params_even_longer",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },

    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(1200e3),
        check_plateau_every_n_steps=int(1200e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(1200e3),
    ),
))

dummy_repeated_rps = PSROScenario(
    name="dummy_repeated_rps_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=RepeatedRPSMultiAgentEnv,
    env_config={
        'fixed_players': True,
        "version": "dummy_repeated_rps",
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,

    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(600e3),
        check_plateau_every_n_steps=int(600e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(600e3),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(dummy_repeated_rps)

scenario_catalog.add(dummy_repeated_rps.with_updates(
    name="dummy_repeated_rps_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))

alphastar_normal_form = PSROScenario(
    name="alphastar_normal_form_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=MatrixGameMultiAgentEnv,
    env_config={
        "version": "AlphaStar.pkl",
        'fixed_players': True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,

    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(600e3),
        check_plateau_every_n_steps=int(600e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(600e3),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(alphastar_normal_form)

scenario_catalog.add(alphastar_normal_form.with_updates(
    name="alphastar_normal_form_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))


blotto_normal_form = PSROScenario(
    name="blotto_normal_form_psro_dqn",
    ray_cluster_cpus=default_if_creating_ray_head(default=8),
    ray_cluster_gpus=default_if_creating_ray_head(default=0),
    ray_object_store_memory_cap_gigabytes=1,
    env_class=MatrixGameMultiAgentEnv,
    env_config={
        "version": "5,3-Blotto.pkl",
        'fixed_players': True,
    },
    mix_metanash_with_uniform_dist_coeff=0.0,
    allow_stochastic_best_responses=False,
    trainer_class=DQNTrainer,
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": SimpleQTorchPolicyPatched,
    },
    num_eval_workers=4,
    games_per_payoff_eval=3000,
    p2sro=False,
    p2sro_payoff_table_exponential_avg_coeff=None,
    p2sro_sync_with_payoff_table_every_n_episodes=None,
    single_agent_symmetric_game=False,
    get_trainer_config=psro_goofspiel_dqn_params,

    psro_get_stopping_condition=lambda: TimeStepsSingleBRRewardPlateauStoppingCondition(
        br_policy_id="best_response",
        dont_check_plateau_before_n_steps=int(600e3),
        check_plateau_every_n_steps=int(600e3),
        minimum_reward_improvement_otherwise_plateaued=0.01,
        max_train_steps=int(600e3),
    ),
    calc_exploitability_for_openspiel_env=True,
)
scenario_catalog.add(blotto_normal_form)

scenario_catalog.add(blotto_normal_form.with_updates(
    name="blotto_normal_form_psro_dqn_avg_pol",
    ray_cluster_gpus=default_if_creating_ray_head(default=1),
    policy_classes={
        "metanash": SimpleQTorchPolicyPatched,
        "best_response": SimpleQTorchPolicyPatched,
        "eval": MixedClassEvalPolicyDQN,
    },
))


