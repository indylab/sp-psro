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
from ray.rllib.utils.typing import PolicyID, ModelWeights, TensorType, AgentID
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch

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
from grl.utils.common import pretty_dict_str, copy_attributes
from grl.utils.port_listings import get_client_port_for_service
from grl.utils.strategy_spec import StrategySpec
from grl.algos.nfsp_rllib.nfsp_torch_avg_policy import NFSPTorchAveragePolicy
from grl.algos.nfsp_rllib.reservoir_replay_buffer import ReservoirReplayActor
from grl.algos.nfsp_rllib.nfsp import NFSPTrainer, get_store_to_avg_policy_buffer_fn, get_avg_policy_buffer_stats_fn

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
                                                                  average_br_reward: float = None,
                                                                  policy_id_to_save: str = "best_response",
                                                                  additional_data=None):
    additional_data_to_save = {
        "policy_num": active_policy_num,
        "timesteps_training_br": timesteps_training_br,
        "episodes_training_br": episodes_training_br,
        "average_br_reward": average_br_reward,
    }
    if additional_data:
        additional_data_to_save.update(additional_data)
    return {
        "checkpoint_path": save_policy_checkpoint(trainer=trainer,
                                                  player=player,
                                                  save_dir=save_dir,
                                                  policy_id_to_save=policy_id_to_save,
                                                  checkpoint_name=f"player_{player}_policy_{active_policy_num}_{policy_id_to_save}",
                                                  additional_data=additional_data_to_save),
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


def disable_learning_for_policy(trainer: Trainer, policy_id: PolicyID):
    target_policy: Policy = trainer.workers.local_worker().policy_map[policy_id]

    # override learning_on_batch to do nothing
    def new_learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        return {}

    target_policy.learn_on_batch = new_learn_on_batch.__get__(target_policy, Policy)


def disable_worker_setting_weights_for_policy(worker: RolloutWorker, policy_id: PolicyID):
    # https://stackoverflow.com/questions/394770/override-a-method-at-instance-level

    def new_set_weights(self, weights: ModelWeights, global_vars: dict = None):
        for pid, w in weights.items():
            if pid == policy_id:
                # don't set weights for target policy_id
                continue
            self.policy_map[pid].set_weights(w)
        if global_vars:
            self.set_global_vars(global_vars)
    worker.set_weights = new_set_weights.__get__(worker, RolloutWorker)
    worker.set_weights_was_modified = True


# def enable_worker_setting_weights(worker: RolloutWorker):
#     if hasattr(worker, "set_weights_was_modified") and worker.set_weights_was_modified:
#         # restore original functionality
#
#         def original_set_weights(self, weights: ModelWeights, global_vars: dict = None):
#             for pid, w in weights.items():
#                 self.policy_map[pid].set_weights(w)
#             if global_vars:
#                 self.set_global_vars(global_vars)
#         worker.set_weights = original_set_weights.__get__(worker, RolloutWorker)
#         worker.set_weights_was_modified = False

@ray.remote(num_cpus=0, num_gpus=0.5)
class PSROBRActor:

    def __init__(self, player: int, results_dir: str, scenario_name: str, psro_manager_port: int,
                 scenario_trainer_config, tmp_env, select_policy,
                 avg_policy_learning_rate, train_avg_policy_for_n_iters_after,
                 psro_manager_host: str, print_train_results=True, previous_br_checkpoint_path=None,
                 end_immediately=False, mwu_window_size=1000,
                 separate_reward_measure_phase=False,
                 metasolver_learning_rate=0.1,
                 disable_sp_br_after_n_steps=None,
                 reset_metasolver_when_sp_br_disabled=False,
                 calculate_openspiel_exploitability_at_end=True):

        other_player = 1 - player
        self.player = player
        self.results_dir = results_dir
        scenario_name = scenario_name
        self.print_train_results = print_train_results
        self.previous_br_checkpoint_path = previous_br_checkpoint_path
        self.end_immediately = end_immediately
        self.separate_reward_measure_phase = separate_reward_measure_phase
        self.disable_sp_br_after_n_steps = disable_sp_br_after_n_steps
        self.reset_metasolver_when_sp_br_disabled = reset_metasolver_when_sp_br_disabled
        self.train_avg_policy_for_n_iters_after = train_avg_policy_for_n_iters_after
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

            # add self play BR
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

        def assert_not_called(agent_id):
            assert False, "This function should never be called."
        avg_policy_model_config = get_trainer_config(tmp_env)["model"]
        avg_policy_model_config['fcnet_hiddens'] = [128, 128, 128]


        avg_trainer_config = {
            "log_level": "DEBUG",
            "framework": "torch",
            "env": scenario.env_class,
            "env_config": scenario.env_config,
            "num_gpus": 0.1,
            "num_gpus_per_worker": 0.0,
            "num_workers": 0,
            "num_envs_per_worker": 1,
            "multiagent": {
                "policies_to_train": ["average_policy"],
                "policies": {
                    "average_policy": (
                        NFSPTorchAveragePolicy, tmp_env.observation_space, tmp_env.action_space, {
                            "model": avg_policy_model_config
                        }),
                },
                "policy_mapping_fn": assert_not_called,
            },
            "learning_starts": 16000,
            "train_batch_size": get_trainer_config(tmp_env)["train_batch_size"] // 2,
            "lr": avg_policy_learning_rate,
            "min_iter_time_s": 0,
        }
        self.avg_trainer = NFSPTrainer(config=avg_trainer_config,
                                        logger_creator=get_trainer_logger_creator(
                                            base_dir=results_dir,
                                            scenario_name=f"{scenario_name}_avg_trainer",
                                            should_log_result_fn=lambda result: result["training_iteration"] % 50 == 0 or "z_metasolver_exploitability" in result))

        get_reservoir_buffer_stats = get_avg_policy_buffer_stats_fn(self.avg_trainer)
        store_to_avg_policy_buffer = get_store_to_avg_policy_buffer_fn(nfsp_trainer=self.avg_trainer)


        class P2SROPreAndPostEpisodeCallbacks(DefaultCallbacks):
            from typing import Dict
            from ray.rllib.env import BaseEnv
            from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
            from ray.rllib.policy import Policy
            from ray.rllib.utils.typing import AgentID, PolicyID
            def on_postprocess_trajectory(self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
                                          agent_id: AgentID, policy_id: PolicyID, policies: Dict[PolicyID, Policy],
                                          postprocessed_batch: SampleBatch,
                                          original_batches: Dict[AgentID, SampleBatch], **kwargs):
                super().on_postprocess_trajectory(worker=worker, episode=episode, agent_id=agent_id,
                                                  policy_id=policy_id, policies=policies,
                                                  postprocessed_batch=postprocessed_batch,
                                                  original_batches=original_batches, **kwargs)

                # make sure this is just one episode
                assert len(set(postprocessed_batch.data["eps_id"])) == 1, len(set(postprocessed_batch.data["eps_id"]))

                if policy_id == "metanash":
                    if "action_probs" in postprocessed_batch:
                        del postprocessed_batch.data["action_probs"]
                    if "behaviour_logits" in postprocessed_batch:
                        del postprocessed_batch.data["behaviour_logits"]

                    br_policy: Policy = policies["metanash"]
                    new_batch = br_policy.postprocess_trajectory(
                        sample_batch=postprocessed_batch,
                        other_agent_batches=original_batches,
                        episode=episode)
                    copy_attributes(src_obj=new_batch, dst_obj=postprocessed_batch)
                    if br_policy.is_sp_br_currently:
                        postprocessed_batch["is_sp_br"] = [True] * postprocessed_batch.count
                    else:
                        postprocessed_batch["is_sp_br"] = [False] * postprocessed_batch.count

                    if br_policy.get_current_policy_id() == "best_response_metanash":
                        assert "exploit_actions" in postprocessed_batch.data, list(postprocessed_batch.keys())
                    else:
                        assert br_policy.get_current_policy_id() == "average_policy_metanash", br_policy.get_current_policy_id()
                        assert "exploit_actions" not in postprocessed_batch.data, list(postprocessed_batch.keys())
                        postprocessed_batch["exploit_actions"] = postprocessed_batch["actions"]

                else:
                    postprocessed_batch["is_sp_br"] = [False] * postprocessed_batch.count

                # else:
                if "q_values" in postprocessed_batch:
                    del postprocessed_batch.data["q_values"]
                if "soft_q_values" in postprocessed_batch:
                    del postprocessed_batch.data["soft_q_values"]
                if "action_probs" in postprocessed_batch:
                    del postprocessed_batch.data["action_probs"]
                if "action_dist_inputs" in postprocessed_batch:
                    del postprocessed_batch.data["action_dist_inputs"]
                if "action_prob" in postprocessed_batch:
                    del postprocessed_batch.data["action_prob"]
                if "action_logp" in postprocessed_batch:
                    del postprocessed_batch.data["action_logp"]

            def on_sample_end(self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs):
                super().on_sample_end(worker=worker, samples=samples, **kwargs)
                assert isinstance(samples, MultiAgentBatch)

                for policy_samples in samples.policy_batches.values():
                    if "action_prob" in policy_samples.data:
                        del policy_samples.data["action_prob"]
                    if "action_logp" in policy_samples.data:
                        del policy_samples.data["action_logp"]

                for policy_id, _policy_samples in samples.policy_batches.items():
                    for episode_samples in _policy_samples.split_by_episode():
                        # make sure this is just one episode
                        assert len(set(episode_samples.data["eps_id"])) == 1, len(set(episode_samples.data["eps_id"]))
                        # make sure this is just one strategy from the metanash
                        assert len(set(episode_samples.data["is_sp_br"])) == 1, len(set(episode_samples.data["is_sp_br"]))

                        if policy_id == "metanash" and episode_samples.data["is_sp_br"][0] == True:
                            new_episode_samples = episode_samples.copy()
                            # Save recorded exploit actions for average policy even if other explore actions were taken.
                            new_episode_samples[SampleBatch.ACTIONS] = new_episode_samples["exploit_actions"]

                            store_to_avg_policy_buffer(MultiAgentBatch(policy_batches={
                                "average_policy": new_episode_samples
                            }, env_steps=new_episode_samples.count))

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

                    sampled_spec_index = ray.get(metasolver.sample_action.remote())

                    # if force_sp_br_play_rate and np.random.random() < force_sp_br_play_rate:
                    #     assert 0.0 < force_sp_br_play_rate < 1.0, force_sp_br_play_rate
                    #     # force using sp_br for some percent of episodes
                    #     sampled_spec_index = len(original_player_probs[other_player])

                    if episode.policy_for(player) == "best_response_non_learning":
                        # force using sp_br when playing non-learning best response opponent
                        sampled_spec_index = len(original_player_probs[other_player])
                    else:
                        assert episode.policy_for(player) == "best_response", episode.policy_for(player)

                    worker.last_sampled_spec_index = sampled_spec_index

                    # enable_worker_setting_weights(worker=worker)

                    if sampled_spec_index == len(original_player_probs[other_player]):
                        # action is self play BR
                        metanash_policy.current_policy.config["explore"] = True
                        try:
                            assert worker.worker_index != 0, "Self-Play PSRO assumes experience workers " \
                                                             "are always remote so that they can have different " \
                                                             "exploration policies than the trainer."
                            metanash_policy.set_current_policy_id("best_response_metanash")
                            metanash_policy.set_weights(worker._latest_self_play_br_weights, is_non_ray_internal_call=True)
                            metanash_policy.is_sp_br_currently = True
                        except AttributeError:
                            print(f"No self play BR weights given to worker. "
                                  f"This is ok if only on the 1st training iteration.")
                            metanash_policy.is_sp_br_currently = False
                    else:
                        new_pure_strat_spec: StrategySpec = opponent_policy_distribution.get_spec_by_index(
                            index=sampled_spec_index)
                        # noinspection PyTypeChecker
                        load_pure_strat(policy=metanash_policy, pure_strat_spec=new_pure_strat_spec, must_have_mixed_class_policy_map_id=True)
                        metanash_policy.is_sp_br_currently = False
                        metanash_policy.current_policy.config["explore"] = scenario.allow_stochastic_best_responses

                    # disabling so the worker can't get weights set during learning in the trainer
                    disable_worker_setting_weights_for_policy(worker=worker, policy_id="metanash")

                    metanash_policy.episodes_since_resample = 1
                elif opponent_policy_distribution is not None:
                    metanash_policy.episodes_since_resample += 1

            def on_train_result(self, *, trainer, result: dict, **kwargs):
                result["scenario_name"] = trainer.scenario_name

                # give workers a copy of the latest self play br weights
                self_play_br_weights = trainer.workers.local_worker().policy_map["metanash"].get_weights()
                def _set_worker_self_play_br_weights(worker: RolloutWorker):
                    worker._latest_self_play_br_weights = self_play_br_weights
                trainer.workers.foreach_worker(_set_worker_self_play_br_weights)

                # match non_learning best response to actual best response
                opponent_br_weights = trainer.workers.local_worker().policy_map["best_response"].get_weights()
                def _set_worker_opponent_br_weights(worker: RolloutWorker):
                    worker.policy_map["best_response_non_learning"].set_weights(opponent_br_weights)
                trainer.workers.foreach_worker(_set_worker_opponent_br_weights)

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

                    ray.get(metasolver.perform_update.remote())

                    result["metasolver_avg_probs"] = ray.get(metasolver.get_avg_weights_action_probs.remote())
                    result["metasolver_latest_iter_probs"] = ray.get(metasolver.get_action_probs.remote())

                if scenario.calc_exploitability_for_openspiel_env and not calculate_openspiel_exploitability_at_end:

                    assert not train_avg_policy_for_n_iters_after, train_avg_policy_for_n_iters_after

                    if result["training_iteration"] % 10 == 0 and metasolver is not None:
                        metanash_probs_0, metanash_probs_1 = trainer.latest_metasolver_probs
                        # metanash_probs_0, metanash_probs_1 = original_player_probs

                        assert len(metanash_probs_1) == len(
                            policy_specs_1) + 1, f"len(metanash_probs_1): {len(metanash_probs_1)}, len(policy_specs_1): {len(policy_specs_1)}"
                        assert len(metanash_probs_0) == len(policy_specs_0) + 1
                        assert len(policy_specs_0) == len(policy_specs_1)

                        policy_specs_0_with_sp_br = [*policy_specs_0, "sp_br_0"]
                        policy_specs_1_with_sp_br = [*policy_specs_1, "sp_br_1"]

                        br_checkpoint_paths = []
                        metanash_weights = []

                        for spec_0, prob_0, spec_1, prob_1 in zip(policy_specs_0_with_sp_br, metanash_probs_0, policy_specs_1_with_sp_br,
                                                                  metanash_probs_1):

                            if spec_0 == "sp_br_0":
                                assert spec_1 == "sp_br_1"
                                br_checkpoint_paths.append(("sp_br_0", "sp_br_1"))
                            else:
                                br_checkpoint_paths.append(
                                    (spec_0.metadata["checkpoint_path"], spec_1.metadata["checkpoint_path"]))
                            metanash_weights.append((prob_0, prob_1))

                        def _load_strat_or_sp_br_weights(rllib_policy, checkpoint_path):
                            if checkpoint_path in ["sp_br_0", "sp_br_1"]:
                                _player_id = int(checkpoint_path[-1])
                                assert _player_id in [0, 1]
                                weights = trainer.latest_avg_sp_br_weights[_player_id]
                                rllib_policy.set_current_policy_id("average_policy_metanash")
                                rllib_policy.set_weights(weights=weights, is_non_ray_internal_call=True)
                            else:
                                load_pure_strat(rllib_policy, checkpoint_path=checkpoint_path, must_have_mixed_class_policy_map_id=True)

                        exploitability = psro_measure_exploitability_nonlstm(
                            br_checkpoint_path_tuple_list=br_checkpoint_paths,
                            metanash_weights=metanash_weights,
                            set_policy_weights_fn=_load_strat_or_sp_br_weights,
                            rllib_policies=[trainer.workers.local_worker().policy_map["metanash_eval"] for _ in range(2)],
                            poker_game_version=tmp_env.game_version,
                            open_spiel_env_config=tmp_env.open_spiel_env_config
                        )
                        result["z_metasolver_exploitability"] = exploitability

                if result["training_iteration"] % 10 == 0:
                    result["reservoir_buffer"] = get_reservoir_buffer_stats()

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
                "policies_to_train": ["metanash", "best_response"],
                "policies": {
                    f"metanash": (
                        scenario.policy_classes["metanash"], tmp_env.observation_space, tmp_env.action_space,
                        {"explore": allow_stochastic_best_response}),

                    f"best_response": (
                        scenario.policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),

                    f"best_response_non_learning": (
                        scenario.policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
                },
                "policy_mapping_fn": select_policy,
            },
        }

        trainer_config["multiagent"]["policies"]["metanash_eval"] = (
                    scenario.policy_classes["eval"], tmp_env.observation_space, tmp_env.action_space,
                    {"explore": allow_stochastic_best_response})

        trainer_config = merge_dicts(trainer_config, scenario_trainer_config)

        self.trainer = trainer_class(config=trainer_config,
                                     logger_creator=get_trainer_logger_creator(
                                         base_dir=results_dir, scenario_name=scenario_name,
                                         should_log_result_fn=lambda result: result["training_iteration"] % 20 == 0 or "z_metasolver_exploitability" in result,
                                         delete_hist_stats=True))

        # scenario_name logged in on_train_result_callback
        self.trainer.scenario_name = scenario_name

        self.trainer.latest_metasolver_probs = None

        def _set_metanash_rollout_policies(worker: RolloutWorker, worker_index: int):
            if worker_index == 0:
                # worker is local, this one needs to be the RL algo's actual policy class to train
                return
            del worker.policy_map["metanash"]
            worker.policy_map["metanash"] = worker.policy_map["metanash_eval"]
        self.trainer.workers.foreach_worker_with_index(_set_metanash_rollout_policies)

        if previous_br_checkpoint_path is not None:
            def _set_br_initial_weights(worker: RolloutWorker):
                br_policy = worker.policy_map["best_response"]
                load_pure_strat(policy=br_policy, checkpoint_path=previous_br_checkpoint_path)
            self.trainer.workers.foreach_worker(_set_br_initial_weights)

        # match non_learning best response to actual best response
        opponent_br_weights = self.trainer.workers.local_worker().policy_map["best_response"].get_weights()
        def _set_worker_opponent_br_weights(worker: RolloutWorker):
            worker.policy_map["best_response_non_learning"].set_weights(opponent_br_weights)
        self.trainer.workers.foreach_worker(_set_worker_opponent_br_weights)

        # Disable setting worker weights so experience gathering can be done
        # by entirely different policies that what the trainer has.
        assert len(self.trainer.workers.remote_workers()) > 0, "This script assumes experience gathering is done by remote workers"
        ray.get([w.apply.remote(lambda worker: disable_worker_setting_weights_for_policy(worker, policy_id="metanash")) for w in self.trainer.workers.remote_workers()])

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
        self._last_metasolver_exploitability = None
        self._latest_local_avg_sp_br_weights = self.avg_trainer.workers.local_worker().policy_map["average_policy"].get_weights()

    def train(self) -> Tuple[Dict, bool]:
        if self.done:
            return {}, self.done

        avg_train_iter_results = {}
        if not self.end_immediately:
            train_iter_results = self.trainer.train()  # do a step (or several) in the main RL loop

            if not self.train_avg_policy_for_n_iters_after:
                avg_train_iter_results = self.avg_trainer.train()
            self.train_iter_count += 1
            avg_br_reward = train_iter_results["policy_reward_mean"]["best_response"]
            avg_metanash_reward = train_iter_results["policy_reward_mean"]["metanash"]
            self.done = False
        else:
            self.done = True
            avg_br_reward = -1
            avg_metanash_reward = -1
            train_iter_results = {}

        self._latest_local_avg_sp_br_weights = self.avg_trainer.workers.local_worker().policy_map["average_policy"].get_weights()

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
            if not self.train_avg_policy_for_n_iters_after:
                self.log(pretty_dict_str(avg_train_iter_results))
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

        if self.disable_sp_br_after_n_steps and self.total_timesteps_training_br >= self.disable_sp_br_after_n_steps:
            disable_learning_for_policy(trainer=self.trainer, policy_id="metanash")
            if self.reset_metasolver_when_sp_br_disabled:
                # reset average policy to quickly adjust to fixed problem
                ray.get(self.metasolver.reset_average_policy.remote())

        if not self.end_immediately and self.stopping_condition.should_stop_this_iter(
                latest_trainer_result=train_iter_results):
            if self.p2sro_manager.can_active_policy_be_set_as_fixed_now(player=self.player,
                                                                        policy_num=self.active_policy_num):
                self.done = True
            else:
                self.log(f"Forcing training to continue since lower policies are still active.")

        if self.done:

            self.log(f"Training stopped. Setting active policy {self.active_policy_num} as fixed.")

            final_policy_metadata = create_metadata_with_new_checkpoint_for_current_best_response(
                trainer=self.trainer, player=self.player, save_dir=checkpoint_dir(trainer=self.trainer),
                timesteps_training_br=0,
                episodes_training_br=0,
                active_policy_num=self.active_policy_num,
                average_br_reward=avg_br_reward,
                additional_data={"mixed_class_policy_map_id": "best_response_metanash"}
            )

            final_policy_metadata["last_metasolver_exploitability"] = None
            final_policy_metadata["mixed_class_policy_map_id"] = "best_response_metanash"

            # submit best response
            opponent_br_spec = self.p2sro_manager.set_active_policy_as_fixed(
                player=self.player, policy_num=self.active_policy_num,
                final_metadata_dict=final_policy_metadata)
            assert opponent_br_spec.pure_strat_index_for_player(self.player) % 2 == 0, opponent_br_spec.pure_strat_index_for_player(self.player)

        return train_iter_results, self.done

    def train_avg_policy(self):
        assert self.done

        assert not hasattr(self.trainer, "latest_avg_sp_br_weights")

        assert self.train_avg_policy_for_n_iters_after, self.train_avg_policy_for_n_iters_after

        self.log(f"Training stopped. "
                 f"Training average policy for {self.train_avg_policy_for_n_iters_after} iterations.")

        for _iteration in range(self.train_avg_policy_for_n_iters_after):
            avg_train_iter_results = self.avg_trainer.train()
            if _iteration % 50 == 0:
                self.log(pretty_dict_str(avg_train_iter_results))


        self._latest_local_avg_sp_br_weights = self.avg_trainer.workers.local_worker().policy_map["average_policy"].get_weights()


    def submit_self_play_policy_and_cleanup(self):
        assert self.done
        assert hasattr(self.trainer, "latest_avg_sp_br_weights")

        ## exploitability
        if scenario.calc_exploitability_for_openspiel_env and self.calculate_openspiel_exploitability_at_end and self.metasolver is not None:
            metanash_probs_0, metanash_probs_1 = self.trainer.latest_metasolver_probs
            # metanash_probs_0, metanash_probs_1 = original_player_probs

            assert len(metanash_probs_1) == len(
                self.policy_specs_1) + 1, f"len(metanash_probs_1): {len(metanash_probs_1)}, len(policy_specs_1): {len(self.policy_specs_1)}"
            assert len(metanash_probs_0) == len(self.policy_specs_0) + 1
            assert len(self.policy_specs_0) == len(self.policy_specs_1)

            policy_specs_0_with_sp_br = [*self.policy_specs_0, "sp_br_0"]
            policy_specs_1_with_sp_br = [*self.policy_specs_1, "sp_br_1"]

            br_checkpoint_paths = []
            metanash_weights = []

            for spec_0, prob_0, spec_1, prob_1 in zip(policy_specs_0_with_sp_br, metanash_probs_0,
                                                      policy_specs_1_with_sp_br,
                                                      metanash_probs_1):

                if spec_0 == "sp_br_0":
                    assert spec_1 == "sp_br_1"
                    br_checkpoint_paths.append(("sp_br_0", "sp_br_1"))
                else:
                    br_checkpoint_paths.append(
                        (spec_0.metadata["checkpoint_path"], spec_1.metadata["checkpoint_path"]))
                metanash_weights.append((prob_0, prob_1))

            def _load_strat_or_sp_br_weights(rllib_policy, checkpoint_path):
                if checkpoint_path in ["sp_br_0", "sp_br_1"]:
                    _player_id = int(checkpoint_path[-1])
                    assert _player_id in [0, 1]
                    weights = self.trainer.latest_avg_sp_br_weights[_player_id]
                    rllib_policy.set_current_policy_id("average_policy_metanash")
                    rllib_policy.set_weights(weights=weights, is_non_ray_internal_call=True)
                else:
                    load_pure_strat(rllib_policy, checkpoint_path=checkpoint_path,
                                    must_have_mixed_class_policy_map_id=True)

            exploitability = psro_measure_exploitability_nonlstm(
                br_checkpoint_path_tuple_list=br_checkpoint_paths,
                metanash_weights=metanash_weights,
                set_policy_weights_fn=_load_strat_or_sp_br_weights,
                rllib_policies=[self.trainer.workers.local_worker().policy_map["metanash_eval"] for _ in range(2)],
                poker_game_version=tmp_env.game_version,
                open_spiel_env_config=tmp_env.open_spiel_env_config
            )
            assert self._last_metasolver_exploitability is None or self._last_metasolver_exploitability == -1, self._last_metasolver_exploitability
            self._last_metasolver_exploitability = exploitability


        # wait for both player policies to be fixed (payoff evals need to be done by manager).
        for player_to_wait_on in range(2):
            wait_count = 0
            wait_amount_s = 2.0
            while True:
                if self.p2sro_manager.is_policy_fixed(player=player_to_wait_on, policy_num=self.active_policy_num):
                    break
                if wait_count % 10 == 0:
                    print(f"Waiting for policy {self.active_policy_num} player {player_to_wait_on} to become fixed a")
                    wait_amount_s *= 2.0
                time.sleep(wait_amount_s)
                wait_count += 1


        # submit self-play best response
        final_self_play_policy_metadata = create_metadata_with_new_checkpoint_for_current_best_response(
            trainer=self.avg_trainer, player=self.other_player, save_dir=checkpoint_dir(trainer=self.avg_trainer),
            timesteps_training_br=self.total_timesteps_training_br,
            episodes_training_br=self.total_episodes_training_br,
            active_policy_num=-1,
            average_br_reward=-1.0,
            policy_id_to_save="average_policy",
            additional_data={"mixed_class_policy_map_id": "average_policy_metanash"}
        )
        final_self_play_policy_metadata["mixed_class_policy_map_id"] = "average_policy_metanash"

        final_self_play_policy_metadata["last_metasolver_exploitability"] = self._last_metasolver_exploitability
        final_self_play_policy_metadata["last_metasolver_probs_for_each_player"] = self.trainer.latest_metasolver_probs

        self_br_final_spec = self.p2sro_manager.claim_new_active_policy_for_player(
            player=self.other_player,
            new_policy_metadata_dict=final_self_play_policy_metadata,
            set_as_fixed_immediately=True)
        assert self_br_final_spec.pure_strat_index_for_player(self.other_player) % 2 == 1, self_br_final_spec.pure_strat_index_for_player(self.other_player)

        if self.active_policy_num > 1:
            # note that self.active_policy_num below is for a different player than the self-play BR
            assert self_br_final_spec.pure_strat_index_for_player(
                self.other_player) == self.active_policy_num + 1, f"self_br_final_spec.pure_strat_index_for_player(self.other_player): {self_br_final_spec.pure_strat_index_for_player(self.other_player)}, self.active_policy_num: {self.active_policy_num}"

        self.trainer.cleanup()
        self.avg_trainer.cleanup()
        time.sleep(10)


    def get_metasolver_probs(self):
        metasolver_probs = ray.get(self.metasolver.get_avg_weights_action_probs.remote())
        assert np.isclose(sum(metasolver_probs), 1.0)
        # if self.add_metanash_action:
        #     metanash_action_prob = metasolver_probs[-1]
        #     metasolver_probs = metasolver_probs[:-1]
        #     assert len(metasolver_probs) == len(self.original_player_metanash_probs[other_player])
        #     metasolver_probs = metasolver_probs + (
        #             metanash_action_prob * self.original_player_metanash_probs[other_player])
        #     assert np.isclose(sum(metasolver_probs), 1.0)
        return metasolver_probs

    def get_original_metanash_probs(self):
        return self.original_player_metanash_probs

    def get_avg_sp_br_weights(self):
        return self._latest_local_avg_sp_br_weights

    def set_avg_sp_br_weights(self, avg_sp_br_weights_for_each_player):
        self.trainer.latest_avg_sp_br_weights = avg_sp_br_weights_for_each_player

    def set_latest_metasolver_probs(self, probs_for_each_player_strat):
        self.trainer.latest_metasolver_probs = probs_for_each_player_strat


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--instant_first_iter', default=False, action='store_true')
    parser.add_argument('--separate_reward_measure_phase', default=False, action='store_true')
    parser.add_argument('--mwu_window_size', default=1000, type=int, required=False)
    parser.add_argument('--mwu_learning_rate', default=0.1, type=float, required=False)
    parser.add_argument('--disable_sp_br_after_n_steps', default=None, type=int, required=False)
    parser.add_argument('--reset_metasolver_when_sp_br_disabled', default=False, action='store_true')
    parser.add_argument('--max_policies', type=int, required=False, default=None)
    parser.add_argument('--avg_policy_learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--force_sp_br_play_rate', type=float, required=False, default=0.05)
    parser.add_argument('--train_avg_policy_for_n_iters_after', type=int, required=False, default=0)
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

    previous_br_checkpoint_path = None
    while True:

        ray_head_address = manager.get_manager_metadata()["ray_head_address"]
        init_ray_for_scenario(scenario=scenario, head_address=ray_head_address, logging_level=logging.INFO)

        learner_references = []

        for _player in range(2):
            other_player = 1 - _player

            def select_policy(agent_id):
                if np.random.random() > commandline_args.force_sp_br_play_rate:
                    if agent_id == _player:
                        return "best_response"
                    elif agent_id == other_player:
                        return "metanash"
                    else:
                        raise ValueError(f"Unknown agent id: {agent_id}")
                else:
                    if agent_id == _player:
                        return "best_response_non_learning"
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
                mwu_window_size=commandline_args.mwu_window_size,
                separate_reward_measure_phase=commandline_args.separate_reward_measure_phase,
                metasolver_learning_rate=float(commandline_args.mwu_learning_rate),
                disable_sp_br_after_n_steps=commandline_args.disable_sp_br_after_n_steps,
                reset_metasolver_when_sp_br_disabled=commandline_args.reset_metasolver_when_sp_br_disabled,
                avg_policy_learning_rate=commandline_args.avg_policy_learning_rate,
                train_avg_policy_for_n_iters_after=commandline_args.train_avg_policy_for_n_iters_after,
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

                if not commandline_args.train_avg_policy_for_n_iters_after:
                    latest_avg_sp_br_weights = []
                    for l in reversed(learner_references):
                        latest_avg_sp_br_weights.append(ray.get(l.get_avg_sp_br_weights.remote()))
                    for l in learner_references:
                        ray.get(l.set_avg_sp_br_weights.remote(latest_avg_sp_br_weights))

            results = ray.get([l.train.remote() for l in learner_references])

            if all(result[1] for result in results):
                # both trainers are done


                if commandline_args.train_avg_policy_for_n_iters_after:
                    assert not commandline_args.calculate_openspiel_exploitability_throughout

                    ray.get([l.train_avg_policy.remote() for l in learner_references])

                    latest_avg_sp_br_weights = []
                    for l in reversed(learner_references):
                        latest_avg_sp_br_weights.append(ray.get(l.get_avg_sp_br_weights.remote()))
                    for l in learner_references:
                        ray.get(l.set_avg_sp_br_weights.remote(latest_avg_sp_br_weights))

                ray.get([l.submit_self_play_policy_and_cleanup.remote() for l in learner_references])

                # wait for both player policies to be fixed (payoff evals need to be done by manager).
                for player_to_wait_on in range(2):
                    wait_count = 0
                    wait_amount_s = 2.0
                    while True:
                        if manager.is_policy_fixed(player=player_to_wait_on, policy_num=psro_iter+1):
                            # ^use psro_iter+1 because we're adding the BR (psro_iter) and self-play BR (psro_iter+1)^
                            break
                        if wait_count % 10 == 0:
                            print(f"Waiting for policy {psro_iter+1} player {player_to_wait_on} to become fixed b")
                            wait_amount_s *= 2.0
                        time.sleep(wait_amount_s)
                        wait_count += 1

                break

        psro_iter += 2

        if commandline_args.max_policies and psro_iter >= commandline_args.max_policies:
            break

    print("Done.")