from typing import List, Optional
from typing import Tuple, Dict, Union

import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType


class FixedActionPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

        if isinstance(action_space, Discrete):
            self.action_type = np.int32
        elif isinstance(action_space, Box):
            self.action_type = np.float32
        else:
            raise NotImplementedError(f"Unkown action space for FixedActionPolicy: {action_space}")

        self.set_fixed_action(action=config.get("action"))

    def set_fixed_action(self, action):
        self.action = action
        if self.action is not None and not (isinstance(self.action, str) and self.action == "random") and action not in self.action_space:
            raise ValueError(f"Action {self.action} isn't in given action space {self.action_space}.")

    def compute_actions(self, obs_batch: Union[List[TensorType], TensorType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorType], TensorType] = None,
                        prev_reward_batch: Union[List[TensorType], TensorType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List[MultiAgentEpisode]] = None, explore: Optional[bool] = None,
                        timestep: Optional[int] = None, **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        if self.action is None:
            raise ValueError("Tried to compute_actions with FixedActionPolicy, but self.action wasn't set.")

        actions = []
        for _ in obs_batch:
            if isinstance(self.action, str) and self.action == "random":
                actions.append(self.action_space.sample())
            else:
                actions.append(self.action)

        return np.asarray(actions, dtype=self.action_type), [], {}

    def compute_single_action(self, obs: TensorType, state: Optional[List[TensorType]] = None,
                              prev_action: Optional[TensorType] = None, prev_reward: Optional[TensorType] = None,
                              info: dict = None, episode: Optional["MultiAgentEpisode"] = None,
                              clip_actions: bool = False, explore: Optional[bool] = None,
                              timestep: Optional[int] = None, **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        out = self.compute_actions(obs_batch=[obs], state_batches=[state], prev_action_batch=prev_action,
                                   prev_reward_batch=[prev_reward], info_batch=[info],
                                   episodes=[episode], explore=explore, timestep=timestep)
        actions, _, action_info = out
        return actions[0], [], {}

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        return {}

    def get_weights(self):
        return {"fixed_action": self.action}

    def set_weights(self, weights):
        self.set_fixed_action(action=weights["fixed_action"])
