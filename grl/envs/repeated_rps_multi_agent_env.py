import copy
import random

import numpy as np
from gym.spaces import Discrete, Box
from open_spiel.python.rl_environment import TimeStep, Environment, StepType, ObservationType
from pyspiel import SpielError
import pyspiel
from pyspiel import GameType

from grl.envs.valid_actions_multi_agent_env import ValidActionsMultiAgentEnv


def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


DEFAULT_CONFIG = {
    'version': "spsro_repeated_rps",
    'fixed_players': True,
    'append_valid_actions_mask_to_obs': True,
}

class RepeatedRPSMultiAgentEnv(ValidActionsMultiAgentEnv):

    def __init__(self, env_config=None):
        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})
        self._fixed_players = env_config['fixed_players']
        self.game_version = env_config['version']
        assert self.game_version in ["spsro_repeated_rps", "spsro_biased_repeated_rps", "dummy_repeated_rps"]
        self._append_valid_actions_mask_to_obs = env_config["append_valid_actions_mask_to_obs"]

        self.openspiel_env = self._get_openspiel_env()
        self.open_spiel_env_config = {
            # "num_repetitions": 4,
            # "enable_infostate": True,
        }

        self.base_num_discrete_actions = int(self.openspiel_env.action_spec()["num_actions"])
        self.num_discrete_actions = int(self.base_num_discrete_actions)
        self._base_action_space = Discrete(self.base_num_discrete_actions)

        self.action_space = Discrete(self.num_discrete_actions)
        self._individual_players_with_continuous_action_space = env_config.get("individual_players_with_continuous_action_space")
        self._individual_players_with_orig_obs_space = env_config.get("individual_players_with_orig_obs_space")

        self.orig_observation_length = self.openspiel_env.observation_spec()["info_state"][0]

        if self._append_valid_actions_mask_to_obs:
            self.observation_length = self.orig_observation_length + self.base_num_discrete_actions
        else:
            self.observation_length = self.orig_observation_length
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.observation_length,))

        self.curr_time_step: TimeStep = None
        self.player_map = None

    def _get_openspiel_env(self):
        stage_game = None
        if self.game_version == "spsro_repeated_rps":
            stage_game = "matrix_rps"
        elif self.game_version == "spsro_biased_repeated_rps":
            stage_game = "biased_matrix_rps"
        elif self.game_version == "dummy_repeated_rps":
            stage_game = "dummy_action_matrix_rps"

        rps_matrix_game = pyspiel.load_game(stage_game)
        # rps_repeated_matrix_game = pyspiel.load_game("repeated_game", {"num_repetitions": 4, "enable_infostate": True, "stage_game": pyspiel.GameParameter(pyspiel.load_game("matrix_rps"))})

        rps_repeated_matrix_game = pyspiel.create_repeated_game(rps_matrix_game,
                                                                {"num_repetitions": 4, "enable_infostate": True, "stage_game": f"{stage_game}()"})
        # rps_repeated_efg = pyspiel.convert_to_turn_based(rps_repeated_matrix_game)

        env = Environment(game=rps_repeated_matrix_game, observation_type=ObservationType.INFORMATION_STATE)
        return env

    def _get_current_obs(self):
        done = self.curr_time_step.last()
        obs = {}
        if done:
            player_ids = [0, 1]
        else:
            curr_player_id = self.curr_time_step.observations["current_player"]
            if curr_player_id == -2:
                player_ids = [0, 1]
            else:
                player_ids = [curr_player_id]

        for player_id in player_ids:
            legal_actions = self.curr_time_step.observations["legal_actions"][player_id]
            legal_actions_mask = np.zeros(self.openspiel_env.action_spec()["num_actions"])
            legal_actions_mask[legal_actions] = 1.0

            info_state = self.curr_time_step.observations["info_state"][player_id]

            force_orig_obs = self._individual_players_with_orig_obs_space is not None and player_id in self._individual_players_with_orig_obs_space

            if self._append_valid_actions_mask_to_obs and not force_orig_obs:
                obs[self.player_map(player_id)] = np.concatenate(
                    (np.asarray(info_state, dtype=np.float32), np.asarray(legal_actions_mask, dtype=np.float32)),
                    axis=0)
            else:
                obs[self.player_map(player_id)] = np.asarray(info_state, dtype=np.float32)

        return obs

    def reset(self):
        """Resets the env and returns observations from ready agents.

        Returns:
            obs (dict): New observations for each ready agent.
        """
        self.curr_time_step = self.openspiel_env.reset()

        if self._fixed_players:
            self.player_map = lambda p: p
        else:
            # swap player mapping in half of the games
            self.player_map = random.choice((lambda p: p,
                                             lambda p: (1 - p)))

        return self._get_current_obs()

    def step(self, action_dict):
        """Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """

        assert 0 in action_dict
        assert 1 in action_dict

        player_actions = []
        for player in [0, 1]:
            legal_actions = self.curr_time_step.observations["legal_actions"][player]

            player_action = action_dict[self.player_map(player)]
            orig_player_action = player_action

            if player_action not in self._base_action_space:
                raise ValueError("Processed player action isn't in the base action space.\n"
                                 f"orig action: {orig_player_action}\n"
                                 f"processed action: {player_action}\n"
                                 f"action space: {self.action_space}\n"
                                 f"base action space: {self._base_action_space}")

            if player_action not in legal_actions:
                if self._illegal_actions_default_to_max_coin_value:
                    max_legal_action = max(legal_actions)
                    assert player_action > max_legal_action, f"player_action: {player_action}, max_legal_action: {max_legal_action}"
                    player_action = max_legal_action
                else:
                    raise ValueError("illegal actions are not allowed")

            player_actions.append(player_action)

        self.curr_time_step = self.openspiel_env.step(player_actions)

        obs = self._get_current_obs()
        done = self.curr_time_step.last()
        dones = {0: done, 1: done, "__all__": done}

        if done:
            rewards = {self.player_map(0): self.curr_time_step.rewards[0],
                       self.player_map(1): self.curr_time_step.rewards[1]}

            assert self.curr_time_step.rewards[0] == -self.curr_time_step.rewards[1]

            infos = {0: {}, 1: {}}

            infos[self.player_map(0)]['game_result_was_invalid'] = False
            infos[self.player_map(1)]['game_result_was_invalid'] = False

            assert sum(
                self.curr_time_step.rewards) == 0.0, "curr_time_step rewards in are terminal state are {} (they should sum to zero)".format(
                self.curr_time_step.rewards)

            infos[self.player_map(0)]['rewards'] = self.curr_time_step.rewards[0]
            infos[self.player_map(1)]['rewards'] = self.curr_time_step.rewards[1]

            if self.curr_time_step.rewards[0] > 0:
                infos[self.player_map(0)]['game_result'] = 'won'
                infos[self.player_map(1)]['game_result'] = 'lost'
            elif self.curr_time_step.rewards[1] > 0:
                infos[self.player_map(1)]['game_result'] = 'won'
                infos[self.player_map(0)]['game_result'] = 'lost'
            else:
                infos[self.player_map(1)]['game_result'] = 'tied'
                infos[self.player_map(0)]['game_result'] = 'tied'
        else:
            # assert self.curr_time_step.rewards[
            #            new_curr_player_id] == 0, "curr_time_step rewards in non terminal state are {}".format(
            #     self.curr_time_step.rewards)
            # assert self.curr_time_step.rewards[-(new_curr_player_id - 1)] == 0

            assert sum(
                self.curr_time_step.rewards) == 0.0, "curr_time_step rewards in state are {} (they should sum to zero)".format(
                self.curr_time_step.rewards)

            rewards = {self.player_map(p): self.curr_time_step.rewards[p] for p in range(2)}

            # assert self.curr_time_step.rewards[1 - new_curr_player_id] == 0.0
            infos = {}

        return obs, rewards, dones, infos


