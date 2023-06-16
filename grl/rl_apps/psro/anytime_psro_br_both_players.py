import argparse
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import ray
from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils import merge_dicts, try_import_torch
from ray.rllib.utils.timer import TimerStat
from grl.algos.p2sro.p2sro_manager import RemoteP2SROManagerClient
from grl.algos.p2sro.p2sro_manager.utils import get_latest_metanash_strategies, PolicySpecDistribution
from grl.rl_apps import GRL_SEED
from grl.rl_apps.psro.general_psro_eval import run_episode
from grl.rl_apps.psro.poker_utils import psro_measure_exploitability_nonlstm
from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rl_apps.scenarios.stopping_conditions import StoppingCondition
from grl.rllib_tools.exp3_solver import MWUSolver
from grl.rllib_tools.policy_checkpoints import save_policy_checkpoint, load_pure_strat
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import pretty_dict_str
from grl.utils.port_listings import get_client_port_for_service
from grl.utils.strategy_spec import StrategySpec

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_checkpoints")


def create_metadata_with_new_checkpoint_for_current_best_response(trainer: Trainer,
                                                                  player: int,
                                                                  save_dir: str,
                                                                  timesteps_training_br: int,
                                                                  episodes_training_br: int,
                                                                  active_policy_num: int = None,
                                                                  average_br_reward: float = None):
    return {
        "checkpoint_path": save_policy_checkpoint(trainer=trainer,
                                                  player=player,
                                                  save_dir=save_dir,
                                                  policy_id_to_save="best_response",
                                                  checkpoint_name=f"player_{player}_policy_{active_policy_num}",
                                                  additional_data={
                                                      "policy_num": active_policy_num,
                                                      "timesteps_training_br": timesteps_training_br,
                                                      "episodes_training_br": episodes_training_br,
                                                      "average_br_reward": average_br_reward,
                                                  }),
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br,
        "average_br_reward": average_br_reward,
    }


def set_best_response_active_policy_spec_and_player_for_all_workers(trainer: Trainer,
                                                                    player: int,
                                                                    active_policy_spec: StrategySpec):
    def _set_policy_spec_on_best_response_policy(worker: RolloutWorker):
        br_policy: Policy = worker.policy_map[f"best_response"]
        br_policy.policy_spec = active_policy_spec
        worker.br_player = player

    trainer.workers.foreach_worker(_set_policy_spec_on_best_response_policy)


def update_all_workers_to_latest_metanash(trainer: Trainer,
                                          br_player: int,
                                          metanash_player: int,
                                          p2sro_manager: RemoteP2SROManagerClient,
                                          active_policy_num: int,
                                          mix_metanash_with_uniform_dist_coeff: float,
                                          one_agent_plays_all_sides: bool = False):
    latest_payoff_table, active_policy_nums, fixed_policy_nums = p2sro_manager.get_copy_of_latest_data()
    latest_strategies: Dict[int, PolicySpecDistribution] = get_latest_metanash_strategies(
        payoff_table=latest_payoff_table,
        as_player=1 if one_agent_plays_all_sides else br_player,
        as_policy_num=active_policy_num,
        fictitious_play_iters=2000,
        mix_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff
    )

    if latest_strategies is None:
        opponent_policy_distribution = None
    else:
        opponent_player = 0 if one_agent_plays_all_sides else metanash_player
        print(f"latest payoff matrix for player {opponent_player}:\n"
              f"{latest_payoff_table.get_payoff_matrix_for_player(player=opponent_player)}")
        print(f"metanash for player {opponent_player}: "
              f"{latest_strategies[opponent_player].probabilities_for_each_strategy()}")

        # get the strategy distribution for the opposing player.
        opponent_policy_distribution = latest_strategies[opponent_player]

        # double check that these policy specs are for the opponent player
        assert opponent_player in opponent_policy_distribution.sample_policy_spec().get_pure_strat_indexes().keys()

    def _set_opponent_policy_distribution_for_worker(worker: RolloutWorker):
        worker.opponent_policy_distribution = opponent_policy_distribution

    trainer.workers.foreach_worker(_set_opponent_policy_distribution_for_worker)


def sync_active_policy_br_and_metanash_with_p2sro_manager(trainer: DQNTrainer,
                                                          player: int,
                                                          metanash_player: int,
                                                          one_agent_plays_all_sides: bool,
                                                          p2sro_manager: RemoteP2SROManagerClient,
                                                          mix_metanash_with_uniform_dist_coeff: float,
                                                          active_policy_num: int,
                                                          timesteps_training_br: int,
                                                          episodes_training_br: int):
    p2sro_manager.submit_new_active_policy_metadata(
        player=player, policy_num=active_policy_num,
        metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=trainer, player=player, save_dir=checkpoint_dir(trainer),
            timesteps_training_br=timesteps_training_br,
            episodes_training_br=episodes_training_br,
            active_policy_num=active_policy_num
        ))

    update_all_workers_to_latest_metanash(p2sro_manager=p2sro_manager, br_player=player,
                                          metanash_player=metanash_player, trainer=trainer,
                                          active_policy_num=active_policy_num,
                                          mix_metanash_with_uniform_dist_coeff=mix_metanash_with_uniform_dist_coeff,
                                          one_agent_plays_all_sides=one_agent_plays_all_sides)


@ray.remote
class PSROBRActor:

    def __init__(self, player: int, results_dir: str, scenario_name: str, psro_manager_port: int,
                 scenario_trainer_config, tmp_env, select_policy,
                 psro_manager_host: str, print_train_results=True, previous_br_checkpoint_path=None,
                 end_immediately=False, add_metanash_action=False, mwu_window_size=1000,
                 separate_reward_measure_phase=False,
                 metasolver_learning_rate=0.1,
                 calculate_openspiel_exploitability_at_end=True):

        other_player = 1 - player
        self.player = player
        self.results_dir = results_dir
        scenario_name = scenario_name
        self.print_train_results = print_train_results
        self.previous_br_checkpoint_path = previous_br_checkpoint_path
        self.end_immediately = end_immediately
        self.add_metanash_action = add_metanash_action
        self.separate_reward_measure_phase = separate_reward_measure_phase
        self.calculate_openspiel_exploitability_at_end = calculate_openspiel_exploitability_at_end

        scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)
        if not isinstance(scenario, PSROScenario):
            raise TypeError(
                f"Only instances of {PSROScenario} can be used here. {scenario.name} is a {type(scenario)}.")

        trainer_class = scenario.trainer_class
        self.single_agent_symmetric_game = scenario.single_agent_symmetric_game
        if self.single_agent_symmetric_game and self.player != 0:
            if self.player is None:
                self.player = 0
            else:
                raise ValueError(f"If treating the game as single agent symmetric, only use player 0 "
                                 f"(one agent plays all sides).")

        self.p2sro = scenario.p2sro
        self.p2sro_sync_with_payoff_table_every_n_episodes = scenario.p2sro_sync_with_payoff_table_every_n_episodes
        self.psro_get_stopping_condition = scenario.psro_get_stopping_condition
        self.mix_metanash_with_uniform_dist_coeff = scenario.mix_metanash_with_uniform_dist_coeff
        allow_stochastic_best_response = scenario.allow_stochastic_best_responses
        should_log_result_fn = scenario.ray_should_log_result_filter

        self.p2sro_manager = RemoteP2SROManagerClient(n_players=2, port=psro_manager_port,
                                                      remote_server_host=psro_manager_host)

        latest_payoff_table, active_policy_nums, fixed_policy_nums = self.p2sro_manager.get_copy_of_latest_data()
        latest_strategies: List[Dict[int, PolicySpecDistribution]] = [get_latest_metanash_strategies(
            payoff_table=latest_payoff_table,
            as_player=p,
            as_policy_num=len(fixed_policy_nums[p]),  # needs to be changed if we have concurrent brs
            fictitious_play_iters=2000,
            mix_with_uniform_dist_coeff=0.0
        ) for p in [1, 0]]
        print(f"self.player: {self.player}, other_player: {other_player} latest strategies: {latest_strategies}")

        if latest_strategies[other_player] is not None:
            original_player_probs = [latest_strategies[p][p].probabilities_for_each_strategy() for p in range(2)]
            self.original_player_metanash_probs = original_player_probs

            metasolver_num_actions = len(fixed_policy_nums[other_player])
            if self.add_metanash_action:
                metasolver_num_actions += 1
                assert metasolver_num_actions == len(original_player_probs[other_player]) + 1, \
                    f"metasolver_num_actions: {metasolver_num_actions}, " \
                    f"original_player_probs[other_player]: {original_player_probs[other_player]}"

            metasolver = MWUSolver.remote(num_actions=metasolver_num_actions,
                                          learning_rate=metasolver_learning_rate,
                                          action_reward_averaging_window=mwu_window_size,
                                          initial_action_probs=None)
            self.metasolver = metasolver

            n_policies = len(fixed_policy_nums[player])
            policy_specs_0 = latest_payoff_table.get_ordered_spec_list_for_player(player=0)[:n_policies]
            policy_specs_1 = latest_payoff_table.get_ordered_spec_list_for_player(player=1)[:n_policies]
            self.policy_specs_0 = policy_specs_0
            self.policy_specs_1 = policy_specs_1
        else:
            metasolver = None
            self.original_player_metanash_probs = None

        self.metasolver = metasolver

        class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):
            from typing import Dict
            from ray.rllib.env import BaseEnv
            from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
            from ray.rllib.policy import Policy

            def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                                 policies: Dict[str, Policy],
                                 episode: MultiAgentEpisode, env_index: int, **kwargs):

                # Sample new pure strategy policy weights from the opponent strategy distribution for the best response to
                # train against. For better runtime performance, this function can be modified to load new weights
                # only every few episodes instead.
                resample_pure_strat_every_n_episodes = 1
                metanash_policy: Policy = policies[f"metanash"]
                opponent_policy_distribution: PolicySpecDistribution = worker.opponent_policy_distribution
                time_for_resample = (not hasattr(metanash_policy, "episodes_since_resample") or
                                     metanash_policy.episodes_since_resample >= resample_pure_strat_every_n_episodes)
                if time_for_resample and opponent_policy_distribution is not None:

                    # sampled_spec_index = np.random.choice(a=list(range(len(worker.metasolver_dist))),
                    #                  p=worker.metasolver_dist)

                    sampled_spec_index = ray.get(metasolver.sample_action.remote())
                    worker.last_sampled_spec_index = sampled_spec_index

                    if sampled_spec_index == len(original_player_probs[other_player]):
                        # action is metanash strat
                        orig_spec_index = sampled_spec_index
                        sampled_spec_index = np.random.choice(np.arange(len(original_player_probs[other_player])),
                                                              p=original_player_probs[other_player])

                    new_pure_strat_spec: StrategySpec = opponent_policy_distribution.get_spec_by_index(
                        index=sampled_spec_index)

                    # new_pure_strat_spec: StrategySpec = opponent_policy_distribution.sample_policy_spec()
                    # noinspection PyTypeChecker
                    load_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec)
                    metanash_policy.episodes_since_resample = 1
                elif opponent_policy_distribution is not None:
                    metanash_policy.episodes_since_resample += 1

            def on_train_result(self, *, trainer, result: dict, **kwargs):
                result["scenario_name"] = trainer.scenario_name

                if result["training_iteration"] % 10 == 0 and metasolver is not None:

                    if separate_reward_measure_phase:
                        local_metanash_policy = trainer.workers.local_worker().policy_map["metanash"]
                        local_br_policy = trainer.workers.local_worker().policy_map["best_response"]
                        opponent_policy_distribution: PolicySpecDistribution = trainer.workers.local_worker().opponent_policy_distribution

                        for action in range(n_policies):
                            print(f"measuring rewards for policy {action}")
                            new_pure_strat_spec: StrategySpec = opponent_policy_distribution.get_spec_by_index(
                                index=action)
                            load_pure_strat(policy=local_metanash_policy, pure_strat_spec=new_pure_strat_spec)
                            policies_for_each_player = [None, None]
                            policies_for_each_player[player] = local_br_policy
                            policies_for_each_player[other_player] = local_metanash_policy
                            for _ in range(mwu_window_size):
                                rewards_for_each_player = run_episode(env=tmp_env,
                                                                      policies_for_each_player=policies_for_each_player)
                                ray.get(metasolver.register_reward_for_action.remote(action=action,
                                                                                     reward=rewards_for_each_player[
                                                                                         other_player]))

                    # metasolver.perform_update.remote(callback=update_worker_dists)

                    ray.get(metasolver.perform_update.remote())

                if scenario.calc_exploitability_for_openspiel_env and not calculate_openspiel_exploitability_at_end:
                    if result["training_iteration"] % 10 == 0 and metasolver is not None:
                        metanash_probs_0, metanash_probs_1 = trainer.latest_metasolver_probs
                        # metanash_probs_0, metanash_probs_1 = original_player_probs

                        assert len(metanash_probs_1) == len(
                            policy_specs_1), f"len(metanash_probs_1): {len(metanash_probs_1)}, len(policy_specs_1): {len(policy_specs_1)}"
                        assert len(metanash_probs_0) == len(policy_specs_0)
                        assert len(policy_specs_0) == len(policy_specs_1)

                        br_checkpoint_paths = []
                        metanash_weights = []

                        for spec_0, prob_0, spec_1, prob_1 in zip(policy_specs_0, metanash_probs_0, policy_specs_1,
                                                                  metanash_probs_1):
                            br_checkpoint_paths.append(
                                (spec_0.metadata["checkpoint_path"], spec_1.metadata["checkpoint_path"]))
                            metanash_weights.append((prob_0, prob_1))

                        exploitability = psro_measure_exploitability_nonlstm(
                            br_checkpoint_path_tuple_list=br_checkpoint_paths,
                            metanash_weights=metanash_weights,
                            set_policy_weights_fn=load_pure_strat,
                            rllib_policies=[trainer.workers.local_worker().policy_map["metanash"] for _ in range(2)],
                            poker_game_version=tmp_env.game_version,
                            open_spiel_env_config=tmp_env.open_spiel_env_config
                        )
                        result["z_metasolver_exploitability"] = exploitability

                super().on_train_result(trainer=trainer, result=result, **kwargs)

            def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                               policies: Dict[str, Policy], episode: MultiAgentEpisode,
                               env_index: int, **kwargs):

                if "zero_sum_rew" in episode.last_info_for(agent_id=worker.br_player):
                    metanash_player = 1 - worker.br_player
                    episode.custom_metrics["zero_sum_reward_best_response"] = \
                    episode.last_info_for(agent_id=worker.br_player)["zero_sum_rew"]
                    episode.custom_metrics["zero_sum_reward_metanash"] = \
                    episode.last_info_for(agent_id=metanash_player)["zero_sum_rew"]
                    episode.custom_metrics["tie_rate"] = episode.last_info_for(agent_id=worker.br_player)["tied"]
                    episode.custom_metrics["win_rate_best_response"] = episode.last_info_for(agent_id=worker.br_player)[
                        "won"]
                    episode.custom_metrics["win_rate_metanash"] = episode.last_info_for(agent_id=metanash_player)["won"]
                    episode.custom_metrics["episode_steps_elapsed"] = episode.last_info_for(agent_id=metanash_player)[
                        "episode_steps_elapsed"]

                if metasolver is not None:
                    ray.get(metasolver.register_reward_for_action.remote(action=worker.last_sampled_spec_index,
                                                                         reward=episode.agent_rewards[
                                                                             (other_player, "metanash")]))

                # If using P2SRO, report payoff results of the actively training BR to the payoff table.
                if not scenario.p2sro:
                    return

                if not hasattr(worker, "p2sro_manager"):
                    worker.p2sro_manager = RemoteP2SROManagerClient(n_players=2,
                                                                    port=psro_manager_port,
                                                                    remote_server_host=psro_manager_host)

                br_policy_spec: StrategySpec = worker.policy_map["best_response"].policy_spec
                if br_policy_spec.pure_strat_index_for_player(player=worker.br_player) == 0:
                    # We're training policy 0 if True (first iteration of PSRO).
                    # The PSRO subgame should be empty, and instead the metanash is a random neural network.
                    # No need to report payoff results for this.
                    return

                # Report payoff results for individual episodes to the p2sro manager to keep a real-time approximation
                # of the payoff matrix entries for (learning) active policies.
                policy_specs_for_each_player: List[StrategySpec] = [None, None]
                payoffs_for_each_player: List[float] = [None, None]
                for (player, policy_name), reward in episode.agent_rewards.items():
                    assert policy_name in ["best_response", "metanash"]
                    policy: Policy = worker.policy_map[policy_name]
                    assert policy.policy_spec is not None
                    policy_specs_for_each_player[player] = policy.policy_spec
                    payoffs_for_each_player[player] = reward
                assert all(payoff is not None for payoff in payoffs_for_each_player)

                # Send payoff result to the manager for inclusion in the payoff table.
                worker.p2sro_manager.submit_empirical_payoff_result(
                    policy_specs_for_each_player=tuple(policy_specs_for_each_player),
                    payoffs_for_each_player=tuple(payoffs_for_each_player),
                    games_played=1,
                    override_all_previous_results=False)

        self.other_player = 1 - player
        self.br_learner_name = f"new_learner_{player}"

        def log(message):
            print(f"({self.br_learner_name}): {message}")

        self.log = log

        trainer_config = {
            "log_level": "INFO",
            "framework": "torch",
            "callbacks": P2SROPreAndPostEpisodeCallbacks,
            "env": scenario.env_class,
            "env_config": scenario.env_config,
            "gamma": 1.0,
            "num_gpus": 0,
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "multiagent": {
                "policies_to_train": [f"best_response"],
                "policies": {
                    f"metanash": (
                        scenario.policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space,
                        {"explore": allow_stochastic_best_response}),
                    f"best_response": (
                        scenario.policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
                },
                "policy_mapping_fn": select_policy,
            },
        }

        trainer_config = merge_dicts(trainer_config, scenario_trainer_config)
        self.trainer = trainer_class(config=trainer_config,
                                     logger_creator=get_trainer_logger_creator(
                                         base_dir=results_dir, scenario_name=scenario_name,
                                         should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0 or "z_metasolver_exploitability" in result))

        # scenario_name logged in on_train_result_callback
        self.trainer.scenario_name = scenario_name

        self.trainer.latest_metasolver_probs = None

        if previous_br_checkpoint_path is not None:
            def _set_br_initial_weights(worker: RolloutWorker):
                br_policy = worker.policy_map["best_response"]
                load_pure_strat(policy=br_policy, checkpoint_path=previous_br_checkpoint_path)

            self.trainer.workers.foreach_worker(_set_br_initial_weights)

        self.active_policy_spec: StrategySpec = self.p2sro_manager.claim_new_active_policy_for_player(
            player=player, new_policy_metadata_dict=create_metadata_with_new_checkpoint_for_current_best_response(
                trainer=self.trainer, player=player, save_dir=checkpoint_dir(self.trainer), timesteps_training_br=0,
                episodes_training_br=0,
                active_policy_num=None
            ))

        self.active_policy_num = self.active_policy_spec.pure_strat_index_for_player(player=player)
        self.br_learner_name = f"policy {self.active_policy_num} player {player}"

        log(f"got policy {self.active_policy_num}")

        set_best_response_active_policy_spec_and_player_for_all_workers(trainer=self.trainer, player=self.player,
                                                                        active_policy_spec=self.active_policy_spec)

        sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=self.trainer,
                                                              player=self.player,
                                                              metanash_player=self.other_player,
                                                              one_agent_plays_all_sides=self.single_agent_symmetric_game,
                                                              p2sro_manager=self.p2sro_manager,
                                                              mix_metanash_with_uniform_dist_coeff=self.mix_metanash_with_uniform_dist_coeff,
                                                              active_policy_num=self.active_policy_num,
                                                              timesteps_training_br=0,
                                                              episodes_training_br=0)

        # Perform main RL training loop. Stop training according to our StoppingCondition.
        self.train_iter_count = 0
        self.episodes_since_last_sync_with_manager = 0
        self.stopping_condition: StoppingCondition = self.psro_get_stopping_condition()

        self.done = False
        self._last_metasolver_exploitability = -1

        self.train_timer = TimerStat(window_size=1000)

        # remote_workers = self.trainer.workers.remote_workers()
        # def update_worker_dists(metasolver: Exp3ModelBasedSolver):
        #     action_probs = metasolver.get_action_probs()
        #
        #     def set_action_probs(worker):
        #         worker.metasolver_dist = action_probs
        #
        #     for w in remote_workers:
        #         w.apply.remote(set_action_probs)
        # ray.get(metasolver.apply_fn.remote(update_worker_dists))

    def train(self) -> Tuple[Dict, bool]:
        if self.done:
            return {}, self.done

        if not self.end_immediately:
            with self.train_timer:
                train_iter_results = self.trainer.train()  # do a step (or several) in the main RL loop
            print(f"Average train iter time: {self.train_timer.mean}")
            self.train_iter_count += 1
            avg_br_reward = train_iter_results["policy_reward_mean"]["best_response"]
            self.done = False
        else:
            self.done = True
            avg_br_reward = -1
            train_iter_results = {}

        if "z_metasolver_exploitability" in train_iter_results:
            self._last_metasolver_exploitability = train_iter_results["z_metasolver_exploitability"]

        if self.print_train_results:
            train_iter_results["p2sro_active_policy_num"] = self.active_policy_num
            train_iter_results["best_response_player"] = self.player
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            try:
                del train_iter_results["info"]["learner"][f"best_response"]["td_error"]
            except KeyError:
                pass

            self.log(pretty_dict_str(train_iter_results))

        self.total_timesteps_training_br = train_iter_results.get("timesteps_total", 0.0)
        self.total_episodes_training_br = train_iter_results.get("episodes_total", 0.0)

        self.episodes_since_last_sync_with_manager += train_iter_results.get("episodes_this_iter", 0.0)
        if self.p2sro and self.episodes_since_last_sync_with_manager >= self.p2sro_sync_with_payoff_table_every_n_episodes:
            if self.p2sro_sync_with_payoff_table_every_n_episodes > 0:
                self.episodes_since_last_sync_with_manager = self.episodes_since_last_sync_with_manager % self.p2sro_sync_with_payoff_table_every_n_episodes
            else:
                self.episodes_since_last_sync_with_manager = 0

            sync_active_policy_br_and_metanash_with_p2sro_manager(trainer=self.trainer,
                                                                  player=self.player,
                                                                  metanash_player=self.other_player,
                                                                  one_agent_plays_all_sides=self.single_agent_symmetric_game,
                                                                  p2sro_manager=self.p2sro_manager,
                                                                  mix_metanash_with_uniform_dist_coeff=self.mix_metanash_with_uniform_dist_coeff,
                                                                  active_policy_num=self.active_policy_num,
                                                                  timesteps_training_br=self.total_timesteps_training_br,
                                                                  episodes_training_br=self.total_episodes_training_br)

        if not self.end_immediately and self.stopping_condition.should_stop_this_iter(
                latest_trainer_result=train_iter_results):
            if self.p2sro_manager.can_active_policy_be_set_as_fixed_now(player=self.player,
                                                                        policy_num=self.active_policy_num):
                self.done = True
            else:
                self.log(f"Forcing training to continue since lower policies are still active.")

        if self.done:
            self.log(f"Training stopped. Setting active policy {self.active_policy_num} as fixed.")

            ## exploitability
            if scenario.calc_exploitability_for_openspiel_env and self.calculate_openspiel_exploitability_at_end and self.metasolver is not None:
                metanash_probs_0, metanash_probs_1 = self.trainer.latest_metasolver_probs
                # metanash_probs_0, metanash_probs_1 = original_player_probs

                assert len(metanash_probs_1) == len(
                    self.policy_specs_1), f"len(metanash_probs_1): {len(metanash_probs_1)}, len(policy_specs_1): {len(self.policy_specs_1)}"
                assert len(metanash_probs_0) == len(self.policy_specs_0)
                assert len(self.policy_specs_0) == len(self.policy_specs_1)

                br_checkpoint_paths = []
                metanash_weights = []

                for spec_0, prob_0, spec_1, prob_1 in zip(self.policy_specs_0, metanash_probs_0, self.policy_specs_1,
                                                          metanash_probs_1):
                    br_checkpoint_paths.append(
                        (spec_0.metadata["checkpoint_path"], spec_1.metadata["checkpoint_path"]))
                    metanash_weights.append((prob_0, prob_1))

                exploitability = psro_measure_exploitability_nonlstm(
                    br_checkpoint_path_tuple_list=br_checkpoint_paths,
                    metanash_weights=metanash_weights,
                    set_policy_weights_fn=load_pure_strat,
                    rllib_policies=[self.trainer.workers.local_worker().policy_map["metanash"] for _ in range(2)],
                    poker_game_version=tmp_env.game_version,
                    open_spiel_env_config=tmp_env.open_spiel_env_config
                )
                self._last_metasolver_exploitability = exploitability

            final_policy_metadata = create_metadata_with_new_checkpoint_for_current_best_response(
                trainer=self.trainer, player=self.player, save_dir=checkpoint_dir(trainer=self.trainer),
                timesteps_training_br=self.total_timesteps_training_br,
                episodes_training_br=self.total_episodes_training_br,
                active_policy_num=self.active_policy_num,
                average_br_reward=avg_br_reward)

            final_policy_metadata["last_metasolver_exploitability"] = self._last_metasolver_exploitability
            final_policy_metadata["last_metasolver_probs_for_each_player"] = self.trainer.latest_metasolver_probs

            self.p2sro_manager.set_active_policy_as_fixed(
                player=self.player, policy_num=self.active_policy_num,
                final_metadata_dict=final_policy_metadata)

            self.trainer.cleanup()
            time.sleep(10)

        return train_iter_results, self.done

    def get_metasolver_probs(self):
        metasolver_probs = ray.get(self.metasolver.get_avg_weights_action_probs.remote())
        assert np.isclose(sum(metasolver_probs), 1.0)
        if self.add_metanash_action:
            metanash_action_prob = metasolver_probs[-1]
            metasolver_probs = metasolver_probs[:-1]
            assert len(metasolver_probs) == len(self.original_player_metanash_probs[other_player])
            metasolver_probs = metasolver_probs + (
                        metanash_action_prob * self.original_player_metanash_probs[other_player])
            assert np.isclose(sum(metasolver_probs), 1.0)
        return metasolver_probs

    def get_original_metanash_probs(self):
        return self.original_player_metanash_probs

    def set_latest_metasolver_probs(self, probs_for_each_player_strat):
        self.trainer.latest_metasolver_probs = probs_for_each_player_strat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--instant_first_iter', default=False, action='store_true')
    parser.add_argument('--add_metanash_action', default=False, action='store_true')
    parser.add_argument('--separate_reward_measure_phase', default=False, action='store_true')
    parser.add_argument('--mwu_window_size', default=1000, type=int, required=False)
    parser.add_argument('--mwu_learning_rate', default=0.1, type=float, required=False)
    parser.add_argument('--calculate_openspiel_exploitability_throughout', action='store_true', default=False)
    parser.add_argument('--psro_port', type=int, required=False, default=None)
    parser.add_argument('--psro_host', type=str, required=False, default='localhost')
    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario
    scenario = scenario_catalog.get(scenario_name=scenario_name)

    get_trainer_config = scenario.get_trainer_config
    tmp_env = scenario.env_class(env_config=scenario.env_config)

    psro_host = commandline_args.psro_host
    psro_port = commandline_args.psro_port
    if psro_port is None:
        psro_port = get_client_port_for_service(service_name=f"seed_{GRL_SEED}_{scenario_name}")

    manager = RemoteP2SROManagerClient(n_players=2, port=os.getenv("P2SRO_PORT", psro_port),
                                       remote_server_host=psro_host)

    manager_log_dir = manager.get_log_dir()

    _, _, fixed_policies_per_player = manager.get_copy_of_latest_data()
    assert len(fixed_policies_per_player[0]) == len(fixed_policies_per_player[1]), fixed_policies_per_player

    psro_iter = len(fixed_policies_per_player[0])
    print(f"\n\n\nStarting on psro iter {psro_iter}\n\n\n")
    print(f"calculate_openspiel_exploitability_throughout: {commandline_args.calculate_openspiel_exploitability_throughout}")

    previous_br_checkpoint_path = None
    while True:

        ray_head_address = manager.get_manager_metadata()["ray_head_address"]
        init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

        learner_references = []

        for _player in range(2):
            other_player = 1 - _player

            def select_policy(agent_id):
                if agent_id == _player:
                    return "best_response"
                elif agent_id == other_player:
                    return "metanash"
                else:
                    raise ValueError(f"Unknown agent id: {agent_id}")

            results_dir = os.path.join(manager_log_dir, f"learners_player_{_player}/")
            print(f"results dir is {results_dir}")

            learner = PSROBRActor.remote(
                player=_player,
                results_dir=results_dir,
                scenario_trainer_config=get_trainer_config(tmp_env),
                tmp_env=tmp_env,
                select_policy=select_policy,
                scenario_name=scenario_name,
                psro_manager_port=psro_port,
                psro_manager_host=psro_host,
                print_train_results=True,
                previous_br_checkpoint_path=previous_br_checkpoint_path,
                end_immediately=commandline_args.instant_first_iter and psro_iter == 0,
                add_metanash_action=commandline_args.add_metanash_action,
                mwu_window_size=commandline_args.mwu_window_size,
                separate_reward_measure_phase=commandline_args.separate_reward_measure_phase,
                metasolver_learning_rate=float(commandline_args.mwu_learning_rate),
                calculate_openspiel_exploitability_at_end=not commandline_args.calculate_openspiel_exploitability_throughout
            )
            learner_references.append(learner)
            time.sleep(5)
        while True:
            if psro_iter > 0:
                latest_metasolver_probs = []
                for l in reversed(learner_references):
                    latest_metasolver_probs.append(ray.get(l.get_metasolver_probs.remote()))

                print(f"metasolver probs: {latest_metasolver_probs}")

                for l in learner_references:
                    print(f"original metanash probs: {ray.get(l.get_original_metanash_probs.remote())}")

                for l in learner_references:
                    ray.get(l.set_latest_metasolver_probs.remote(latest_metasolver_probs))

            results = ray.get([l.train.remote() for l in learner_references])

            if all(result[1] for result in results):
                # both trainers are done

                # wait for both player policies to be fixed (payoff evals need to be done by manager).
                for player_to_wait_on in range(2):
                    wait_count = 0
                    while True:
                        if manager.is_policy_fixed(player=player_to_wait_on, policy_num=psro_iter):
                            break
                        if wait_count % 10 == 0:
                            print(f"Waiting for policy {psro_iter} player {player_to_wait_on} to become fixed")
                        time.sleep(2.0)
                        wait_count += 1

                break

        psro_iter += 1
