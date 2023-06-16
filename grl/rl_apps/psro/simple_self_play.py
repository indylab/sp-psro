import argparse
import logging
import os
from typing import Type, Dict

import termcolor
from termcolor import colored

from ray.rllib.agents import Trainer
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy import Policy
from ray.rllib.utils import merge_dicts, try_import_torch

from grl.rl_apps.scenarios.catalog import scenario_catalog
from grl.rl_apps.scenarios.psro_scenario import PSROScenario
from grl.rl_apps.scenarios.ray_setup import init_ray_for_scenario
from grl.rllib_tools.policy_checkpoints import save_policy_checkpoint, load_pure_strat
from grl.rllib_tools.space_saving_logger import get_trainer_logger_creator
from grl.utils.common import pretty_dict_str, ensure_dir, data_dir
from grl.utils.strategy_spec import StrategySpec
from grl.rl_apps.nfsp.openspiel_utils import nfsp_measure_exploitability_nonlstm

torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_policy_checkpoints")


def spec_checkpoint_dir(trainer: Trainer):
    return os.path.join(trainer.logdir, "br_policy_checkpoint_specs")


def create_metadata_with_new_checkpoint(br_trainer: Trainer,
                                        policy_id_to_save: str,
                                        policy_player: int,
                                        save_dir: str,
                                        timesteps_training: int,
                                        episodes_training: int,
                                        checkpoint_name=None
                                        ):
    return {
        "checkpoint_path": save_policy_checkpoint(trainer=br_trainer,
                                                               policy_id_to_save=policy_id_to_save,
                                                               save_dir=save_dir,
                                                               player=policy_player,
                                                               additional_data={
                                                                   "timesteps_training": timesteps_training,
                                                                   "episodes_training": episodes_training
                                                               },
                                                               checkpoint_name=checkpoint_name),
        "timesteps_training": timesteps_training,
        "episodes_training": episodes_training
    }


def train_self_play(results_dir, scenario_name, print_train_results=True, restore_path=None):

    if restore_path is not None:
        checkpoint_specs_dir = restore_path
        player_checkpoint_specs = [None, None]
        player_checkpoint_iters = [None, None]
        for filename in os.listdir(checkpoint_specs_dir):
            if filename.endswith(".txt"):
                continue
            _, _, _, player_str, _, iter_str = filename.split("_")
            iter = int(iter_str.replace(".json", ""))
            player = int(player_str)
            print(f"player: {player} iter: {iter}, {filename}")

            spec = StrategySpec.from_json_file(json_file_path=os.path.join(checkpoint_specs_dir, filename))
            print(spec.metadata)

            if player_checkpoint_iters[player] is None or iter > player_checkpoint_iters[player]:
                player_checkpoint_specs[player] = spec
                player_checkpoint_iters[player] = iter
        assert all(e is not None for e in player_checkpoint_specs)
        assert all(e is not None for e in player_checkpoint_iters)
        assert len(set(player_checkpoint_iters)) == 1
        restore_num_timesteps = player_checkpoint_specs[0].metadata["timesteps_training"]
        restore_num_episodes = player_checkpoint_specs[0].metadata["episodes_training"]
        restore_iter = player_checkpoint_iters[0]

        print(termcolor.colored(f"restored to iter {restore_iter}, "
                                f"timesteps: {restore_num_timesteps}, "
                                f"episodes: {restore_num_episodes}", "green"))

    else:
        restore_num_timesteps = 0
        restore_num_episodes = 0
        restore_iter = 0
        player_checkpoint_specs = None

    scenario: PSROScenario = scenario_catalog.get(scenario_name=scenario_name)

    env_class = scenario.env_class
    env_config = scenario.env_config
    trainer_class = scenario.trainer_class
    policy_classes: Dict[str, Type[Policy]] = scenario.policy_classes
    single_agent_symmetric_game = scenario.single_agent_symmetric_game
    if single_agent_symmetric_game:
        raise NotImplementedError

    get_trainer_config = scenario.get_trainer_config
    should_log_result_fn = scenario.ray_should_log_result_filter
    calculate_openspiel_metanash = scenario.calc_exploitability_for_openspiel_env

    def _create_env():
        return env_class(env_config=env_config)

    checkpoint_every_n_iters = 500

    class PreAndPostEpisodeCallbacks(DefaultCallbacks):

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            result["scenario_name"] = trainer.scenario_name
            training_iteration = result["training_iteration"]
            super().on_train_result(trainer=trainer, result=result, **kwargs)

            if (calculate_openspiel_metanash and
                    (training_iteration == 1 or training_iteration % 100 == 0)):
                base_env = _create_env()
                open_spiel_env_config = base_env.open_spiel_env_config
                openspiel_game_version = base_env.game_version
                local_br_policy_0 = trainer.workers.local_worker().policy_map["best_response_0"]
                local_br_policy_1 = trainer.workers.local_worker().policy_map["best_response_1"]
                exploitability = nfsp_measure_exploitability_nonlstm(
                    rllib_policies=[local_br_policy_0, local_br_policy_1],
                    poker_game_version=openspiel_game_version,
                    open_spiel_env_config=open_spiel_env_config
                )
                result["avg_policy_exploitability"] = exploitability
                logger.info(colored(
                    f"(Graph this in a notebook) Exploitability: {exploitability} - Saving exploitability stats "
                    f"to {os.path.join(trainer.logdir, 'result.json')}", "green"))

            if training_iteration % checkpoint_every_n_iters == 0 or training_iteration == 1:

                training_iteration += restore_iter

                for player in range(2):
                    checkpoint_metadata = create_metadata_with_new_checkpoint(
                        policy_id_to_save=f"best_response_{player}",
                        br_trainer=trainer,
                        policy_player=player,
                        save_dir=checkpoint_dir(trainer=trainer),
                        timesteps_training=result["timesteps_total"] + restore_num_timesteps,
                        episodes_training=result["episodes_total"] + restore_num_episodes,
                        checkpoint_name=f"best_response_player_{player}_iter_{training_iteration}.h5"
                    )
                    joint_pol_checkpoint_spec = StrategySpec(
                        strategy_id=f"best_response_player_{player}_iter_{training_iteration}",
                        metadata=checkpoint_metadata)
                    checkpoint_path = os.path.join(spec_checkpoint_dir(trainer),
                                                   f"best_response_player_{player}_iter_{training_iteration}.json")
                    ensure_dir(checkpoint_path)
                    with open(checkpoint_path, "+w") as checkpoint_spec_file:
                        checkpoint_spec_file.write(joint_pol_checkpoint_spec.to_json())

    def select_policy(agent_id):
        if agent_id == 0:
            return "best_response_0"
        elif agent_id == 1:
            return "best_response_1"
        else:
            raise ValueError(f"Unknown agent id: {agent_id}")

    init_ray_for_scenario(scenario=scenario, head_address=None, logging_level=logging.INFO)

    tmp_env = env_class(env_config=env_config)

    trainer_config = {
        "log_level": "DEBUG",
        "callbacks": PreAndPostEpisodeCallbacks,
        "env": env_class,
        "env_config": env_config,
        "gamma": 1.0,
        "num_gpus": 0,
        "num_workers": 0,
        "num_envs_per_worker": 1,
        "multiagent": {
            "policies_to_train": [f"best_response_0", "best_response_1"],
            "policies": {
                f"best_response_0": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
                f"best_response_1": (
                policy_classes["best_response"], tmp_env.observation_space, tmp_env.action_space, {}),
            },
            "policy_mapping_fn": select_policy,
        },
    }

    trainer_config = merge_dicts(trainer_config, get_trainer_config(tmp_env))



    trainer = trainer_class(config=trainer_config,
                            logger_creator=get_trainer_logger_creator(
                                base_dir=results_dir, scenario_name=scenario_name,
                                should_log_result_fn=should_log_result_fn))

    if player_checkpoint_specs is not None:
        def restore_br_weights(worker):
            for _p in range(2):
                print(termcolor.colored(f"Restoring spec: {player_checkpoint_specs[_p].to_json()}", "green"))
                load_pure_strat(policy=worker.policy_map[f"best_response_{_p}"],
                                pure_strat_spec=player_checkpoint_specs[_p])
        trainer.workers.foreach_worker(restore_br_weights)

    # scenario_name logged in on_train_result_callback
    trainer.scenario_name = scenario_name

    # Perform main RL training loop.
    while True:
        train_iter_results = trainer.train()  # do a step (or several) in the main RL loop

        if print_train_results:
            # Delete verbose debugging info before printing
            if "hist_stats" in train_iter_results:
                del train_iter_results["hist_stats"]
            for key in ["best_response_0", "best_response_1"]:
                if "td_error" in train_iter_results["info"]["learner"][key]:
                    del train_iter_results["info"]["learner"][key]["td_error"]
            print(pretty_dict_str(train_iter_results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str)

    # only restores weights, no change to iter/timestep/episode counting
    parser.add_argument('--restore', type=str, default=None)

    commandline_args = parser.parse_args()

    scenario_name = commandline_args.scenario

    results_dir = os.path.join(data_dir(), f"self_play_{scenario_name}")
    print(f"results dir is {results_dir}")

    train_self_play(
        results_dir=results_dir,
        scenario_name=scenario_name,
        print_train_results=True,
        restore_path=commandline_args.restore
    )
