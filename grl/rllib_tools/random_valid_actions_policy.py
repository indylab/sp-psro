from typing import List, Optional
from typing import Tuple, Dict, Union

import numpy as np
from gym.spaces import Box, Discrete
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorType


class RandomValidActionPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

        if isinstance(action_space, Discrete):
            self.action_type = np.int32
        else:
            raise NotImplementedError(f"Unsupported action space for FixedActionPolicy: {action_space}")

        self.orig_observation_length = config["orig_observation_length"]
        self.observation_length = observation_space.shape[0]

        self.always_use_first_legal_action = bool(config.get("always_use_first_legal_action", False))

    def compute_actions(self, obs_batch: Union[List[TensorType], TensorType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorType], TensorType] = None,
                        prev_reward_batch: Union[List[TensorType], TensorType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List[MultiAgentEpisode]] = None, explore: Optional[bool] = None,
                        timestep: Optional[int] = None, **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        actions = []
        for obs in obs_batch:
            legal_actions_mask = obs[self.orig_observation_length:]
            legal_actions = np.ravel(np.argwhere(legal_actions_mask == 1.0))

            if self.always_use_first_legal_action:
                actions.append(legal_actions[0])
            else:
                actions.append(np.random.choice(legal_actions))
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
        return {}

    def set_weights(self, weights):
        return

