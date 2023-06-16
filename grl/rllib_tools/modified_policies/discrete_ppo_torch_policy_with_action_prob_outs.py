import logging
from typing import Dict, List

from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, vf_preds_fetches
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


def action_outs(policy: Policy, input_dict: Dict[str, TensorType],
                state_batches: List[TensorType], model: ModelV2,
                action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    fetches = vf_preds_fetches(policy=policy, input_dict=input_dict, state_batches=state_batches,
                               model=model, action_dist=action_dist)
    categorical_action_dict: torch.distributions.categorical.Categorical = action_dist.dist
    fetches["action_probs"] = categorical_action_dict.probs
    return fetches


DiscretePPOTorchPolicyWithActionProbsOuts = PPOTorchPolicy.with_updates(
    name="DiscretePPOTorchPolicyWithActionProbsOuts",
    extra_action_out_fn=action_outs,
)
