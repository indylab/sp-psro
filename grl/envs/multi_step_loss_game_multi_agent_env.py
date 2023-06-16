from grl.envs.multi_dim_loss_game_alpha_multi_agent_env import LossGameMultiDimMultiAgentEnv

import numpy as np
from gym.spaces import Discrete, Box


class MultiStepLossGameMultiDimMultiAgentEnv(LossGameMultiDimMultiAgentEnv):

    def __init__(self, env_config=None):
        super().__init__(env_config)

        assert len(self.discrete_actions_for_players) == 0
        self._base_action_space = self.action_space
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = Box(low=-500.0, high=500.0, shape=(self.dim + 1 + 1,), dtype=np.float32)

        self._action_buffers = None
        self._current_obs = None

    def _obs_with_percent_repeat(self, obs_dict: dict):
        new_obs = {}
        for agent_id, obs in obs_dict.items():
            action_buffer_percent = len(self._action_buffers[agent_id]) / self.dim
            new_obs[agent_id] = np.concatenate(([action_buffer_percent], obs))
        return new_obs

    def reset(self):
        self._action_buffers = [[], []]
        self._current_obs = super().reset()

        return self._obs_with_percent_repeat(self._current_obs)

    def step(self, action_dict):

        assert len(action_dict) == 2
        for i in [0, 1]:
            assert action_dict[i] in self.action_space

        for agent_id, action in action_dict.items():
            assert len(action) == 1
            self._action_buffers[agent_id].append(action[0])

        assert len(self._action_buffers[0]) == len(self._action_buffers[1])

        if len(self._action_buffers[0]) % self.dim == 0:
            base_action_dict = {i: np.asarray(self._action_buffers[i]) for i in [0, 1]}
            for i in [0, 1]:
                assert base_action_dict[i] in self._base_action_space, (base_action_dict, self._base_action_space)
            self._current_obs, rews, dones, info = super().step(base_action_dict)
            self._action_buffers = [[], []]

            return self._obs_with_percent_repeat(obs_dict=self._current_obs), rews, dones, info

        rews = {i: 0.0 for i in self._current_obs.keys()}
        dones = {i: False for i in self._current_obs.keys()}
        dones["__all__"] = False
        info = {}
        return self._obs_with_percent_repeat(obs_dict=self._current_obs), rews, dones, info
