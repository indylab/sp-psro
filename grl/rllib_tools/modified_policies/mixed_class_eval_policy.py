from typing import Dict, Union, List, Optional, Tuple

from copy import deepcopy
import gym
from ray.rllib import Policy, SampleBatch
from ray.rllib.utils.typing import ModelWeights, TensorType, TrainerConfigDict
from grl.rllib_tools.modified_policies.simple_q_torch_policy import SimpleQTorchPolicyPatched, SimpleQTorchPolicyPatchedSoftMaxSampling
from grl.algos.nfsp_rllib.nfsp_torch_avg_policy import NFSPTorchAveragePolicy
from grl.rllib_tools.modified_policies.discrete_ppo_torch_policy_with_action_prob_outs import DiscretePPOTorchPolicyWithActionProbsOuts
class MixedClassEvalPolicy(Policy):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        super().__init__(observation_space, action_space, config)
        raise NotImplementedError

    def set_current_policy_id(self, policy_id: str):
        raise NotImplementedError


class MixedClassEvalPolicyDQN(MixedClassEvalPolicy):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        model_config = deepcopy(config["model"])

        model_config['fcnet_hiddens'] = [128, 128, 128]

        self._policy_map = {
            "best_response_metanash": SimpleQTorchPolicyPatched(observation_space, action_space, config),
            "average_policy_metanash": NFSPTorchAveragePolicy(observation_space, action_space, {"model": model_config, "framework": "torch"})
        }

        self._current_policy_id = "average_policy_metanash"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]

    def set_current_policy_id(self, policy_id: str):
        self._current_policy_id = policy_id
        self.current_policy = self._policy_map[self._current_policy_id]

    def get_current_policy_id(self):
        return self._current_policy_id

    def get_initial_state(self) -> List[TensorType]:
        return self.current_policy.get_initial_state()

    def compute_actions(self, *args, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return self.current_policy.compute_actions(*args, **kwargs)

    def compute_single_action(self, *args, **kwargs) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return self.current_policy.compute_single_action(*args, **kwargs)

    def learn_on_batch(self, samples: SampleBatch) -> Dict[str, TensorType]:
        raise NotImplementedError

    def get_weights(self) -> ModelWeights:
        return self.current_policy.get_weights()

    def set_weights(self, weights: ModelWeights, is_non_ray_internal_call=False) -> None:
        if is_non_ray_internal_call:
            self.current_policy.set_weights(weights)

    def is_recurrent(self) -> bool:
        return self.current_policy.is_recurrent()


class MixedClassEvalPolicyDQNOld(MixedClassEvalPolicyDQN):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        model_config = deepcopy(config["model"])

        self._policy_map = {
            "best_response_metanash": SimpleQTorchPolicyPatched(observation_space, action_space, config),
            "average_policy_metanash": NFSPTorchAveragePolicy(observation_space, action_space, {"model": model_config, "framework": "torch"})
        }

        self._current_policy_id = "average_policy_metanash"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]


class MixedClassEvalPolicyDQNDebug(MixedClassEvalPolicyDQN):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        self._policy_map = {
            "best_response_metanash": SimpleQTorchPolicyPatched(observation_space, action_space, config),
            "sp_br_sanity_check": SimpleQTorchPolicyPatched(observation_space, action_space, config),
        }

        self._current_policy_id = "sp_br_sanity_check"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]


class MixedClassEvalPolicyDQNDebugSameInstance(MixedClassEvalPolicyDQN):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        self._policy_map = {
            "best_response_metanash": SimpleQTorchPolicyPatched(observation_space, action_space, config),
        }
        self._policy_map["sp_br_sanity_check"] = self._policy_map["best_response_metanash"]

        self._current_policy_id = "sp_br_sanity_check"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]


class MixedClassEvalPolicyDQNSoftmax(MixedClassEvalPolicyDQN):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        model_config = deepcopy(config["model"])

        model_config['fcnet_hiddens'] = [128, 128, 128]

        self._policy_map = {
            "best_response_metanash": SimpleQTorchPolicyPatchedSoftMaxSampling(observation_space, action_space, config),
            "average_policy_metanash": NFSPTorchAveragePolicy(observation_space, action_space, {"model": model_config, "framework": "torch"})
        }

        self._current_policy_id = "average_policy_metanash"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]


class MixedClassEvalPolicyPPO(MixedClassEvalPolicyDQN):

    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, config: TrainerConfigDict):
        Policy.__init__(self, observation_space, action_space, config)

        model_config = deepcopy(config["model"])

        model_config['fcnet_hiddens'] = [128, 128, 128]

        self._policy_map = {
            "best_response_metanash": DiscretePPOTorchPolicyWithActionProbsOuts(observation_space, action_space, config),
            "average_policy_metanash": NFSPTorchAveragePolicy(observation_space, action_space, {"model": model_config, "framework": "torch"})
        }

        self._current_policy_id = "average_policy_metanash"
        self.current_policy: Policy = self._policy_map[self._current_policy_id]