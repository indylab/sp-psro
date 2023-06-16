import copy
import random
import os
import numpy as np
from gym.spaces import Discrete, Box
from open_spiel.python.rl_environment import TimeStep, Environment, StepType, ObservationType
from pyspiel import SpielError
import pyspiel
from pyspiel import GameType
from ray.rllib.env import MultiAgentEnv
from grl.utils.common import assets_dir
from grl.envs.valid_actions_multi_agent_env import ValidActionsMultiAgentEnv
import pickle

def with_base_config(base_config, extra_config):
    config = copy.deepcopy(base_config)
    config.update(extra_config)
    return config


def get_openspiel_game_from_normal_from_game_asset_id(game_asset_id: str):
    with open(os.path.join(assets_dir(), "normal_form_games", f"{game_asset_id}"), "rb") as fh:
        payoffs = np.asarray(pickle.load(fh))
    return pyspiel.create_matrix_game(payoffs, -payoffs)

DEFAULT_CONFIG = {
    'version': "5,3-Blotto.pkl",
    'fixed_players': True,
}

class MatrixGameMultiAgentEnv(MultiAgentEnv):

    def __init__(self, env_config=None):
        env_config = with_base_config(base_config=DEFAULT_CONFIG, extra_config=env_config if env_config else {})
        self._fixed_players = env_config['fixed_players']
        self.game_version = env_config['version']
        self._append_valid_actions_mask_to_obs = False

        self.openspiel_env = self._get_openspiel_env()
        self.open_spiel_env_config = {

        }

        self.base_num_discrete_actions = int(self.openspiel_env.action_spec()["num_actions"])
        self.num_discrete_actions = int(self.base_num_discrete_actions)
        self._base_action_space = Discrete(self.base_num_discrete_actions)

        self.action_space = Discrete(self.num_discrete_actions)

        self.orig_observation_length = self.openspiel_env.observation_spec()["info_state"][0]

        if self._append_valid_actions_mask_to_obs:
            self.observation_length = self.orig_observation_length + self.base_num_discrete_actions
        else:
            self.observation_length = self.orig_observation_length
        self.observation_space = Box(low=0.0, high=1.0, shape=(self.observation_length,))

        self.curr_time_step: TimeStep = None
        self.player_map = None

    def _get_openspiel_env(self):
        game = get_openspiel_game_from_normal_from_game_asset_id(self.game_version)
        env = Environment(game=game, observation_type=ObservationType.INFORMATION_STATE)
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


            if self._append_valid_actions_mask_to_obs:
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

    def __getstate__(self):
        # capture what is normally pickled (Openspiel envs dont place nice with pickle)
        state = self.__dict__.copy()
        # replace the `value` key:
        state['openspiel_env'] = None
        # what we return here will be stored in the pickle
        return state

    def __setstate__(self, newstate):
        # re-create the instance
        game = get_openspiel_game_from_normal_from_game_asset_id(newstate['game_version'])
        env = Environment(game=game, observation_type=ObservationType.INFORMATION_STATE)

        newstate['openspiel_env'] = env

        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)

