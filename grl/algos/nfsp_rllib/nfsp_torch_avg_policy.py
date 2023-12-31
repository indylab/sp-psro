import logging
from typing import Dict, Tuple, Type, List

import gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_action_dist import (Categorical)
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.schedules import ConstantSchedule, PiecewiseSchedule
from ray.rllib.utils.typing import TensorType, TrainerConfigDict

import grl
from grl.rllib_tools.modified_policies.safe_set_weights_policy_mixin import SafeSetWeightsPolicyMixin

AVG_POL_SCOPE = "avg_pol"

torch, nn = try_import_torch()

import torch
import torch.nn as nn
import torch.nn.functional as F

# F = None
# if nn:
#     F = nn.functional
logger = logging.getLogger(__name__)


def compute_policy_logits(policy: Policy,
                          model: ModelV2,
                          obs: TensorType,
                          is_training=None) -> TensorType:
    model_out, _ = model({
        SampleBatch.CUR_OBS: obs,
        "is_training": is_training
        if is_training is not None else policy._get_is_training_placeholder(),
    }, [], None)

    return model_out


def get_distribution_inputs_and_class(
        policy: Policy,
        model: ModelV2,
        obs_batch: TensorType,
        *,
        is_training=True,
        **kwargs) -> Tuple[TensorType, type, List[TensorType]]:
    """Build the action distribution"""
    logits = compute_policy_logits(policy, model, obs_batch, is_training)
    logits = logits[0] if isinstance(logits, tuple) else logits

    policy.logits = logits
    return policy.logits, (TorchCategorical
                           if policy.config["framework"] == "torch" else
                           Categorical), []  # state-outs


def build_avg_model_and_distribution(
        policy: Policy, obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(f"Action space {action_space} is not supported for NFSP.")

    policy.avg_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n,
        model_config=config["model"],
        framework=config["framework"],
        name=AVG_POL_SCOPE)

    policy.avg_func_vars = policy.avg_model.variables()

    return policy.avg_model, TorchCategorical


# def soft_cross_entropy(input, target):
#     # TODO double check this
#     # https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch
#     logprobs = torch.nn.functional.log_softmax(input, dim=1)
#     return -(target * logprobs).sum() / input.shape[0]

def build_supervised_learning_loss(policy: Policy, model: ModelV2, dist_class: Type[TorchDistributionWrapper],
                                   train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for SimpleQTorchPolicy.

    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]): The action distribution class.
        train_batch (SampleBatch): The training data.

    Returns:
        TensorType: A single loss tensor.
    """
    logits_t = compute_policy_logits(policy=policy,
                                     model=policy.avg_model,
                                     obs=train_batch[SampleBatch.CUR_OBS],
                                     is_training=True)

    action_targets_t = train_batch[SampleBatch.ACTIONS].long()

    # policy.loss = soft_cross_entropy(input=logits_t, target=action_targets_t)

    policy.loss = F.cross_entropy(input=logits_t, target=action_targets_t)
    return policy.loss


def behaviour_logits_fetches(
        policy: Policy, input_dict: Dict[str, TensorType],
        state_batches: List[TensorType], model: ModelV2,
        action_dist: TorchDistributionWrapper) -> Dict[str, TensorType]:
    """Defines extra fetches per action computation.

    Args:
        policy (Policy): The Policy to perform the extra action fetch on.
        input_dict (Dict[str, TensorType]): The input dict used for the action
            computing forward pass.
        state_batches (List[TensorType]): List of state tensors (empty for
            non-RNNs).
        model (ModelV2): The Model object of the Policy.
        action_dist (TorchDistributionWrapper): The instantiated distribution
            object, resulting from the model's outputs and the given
            distribution class.

    Returns:
        Dict[str, TensorType]: Dict with extra tf fetches to perform per
            action computation.
    """
    return {
        "action_probs": policy.action_probs,
        "behaviour_logits": policy.logits,
    }


def action_sampler(policy, model, input_dict, state, explore, timestep):
    obs: np.ndarray = input_dict['obs']

    # assert not np.isnan(obs).any(), f"obs contained NaN:\n{obs}"

    is_training = False
    logits = compute_policy_logits(policy, model, obs, is_training)
    logits = logits[0] if isinstance(logits, tuple) else logits

    # assert not torch.isnan(logits).any(), "action logits contained NaN"

    # action_probs_batch = F.softmax(logits, dim=1)
    action_log_probs_batch = F.log_softmax(logits, dim=1)
    # todo, getting NaNs. probably change to https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html
    # assert not torch.isnan(action_log_probs_batch).any(), f"action_log_probs_batch contained NaN, logits:\n{logits}\nobs:\n{obs}"


    policy.logits = logits
    policy.action_probs = torch.exp(action_log_probs_batch)
    # print(f"probs: {action_probs_batch}")
    # assert not torch.isnan(policy.action_probs).any(), f"policy.action_probs contained NaN"

    actions = []
    logps = []
    for action_probs, action_log_probs in zip(policy.action_probs.cpu().detach().numpy(), action_log_probs_batch.cpu().detach().numpy()):
        action = np.random.choice(range(0, len(action_probs)), p=action_probs)
        logp = action_log_probs[action]
        # print(f"action: {action}, logp: {logp}")
        actions.append(action)
        logps.append(logp)
    state_out = state
    return np.asarray(actions, dtype=np.int32), None, state_out


def sgd_optimizer(policy: Policy,
                  config: TrainerConfigDict) -> "torch.optim.Optimizer":
    return torch.optim.SGD(
        policy.avg_func_vars, lr=policy.config["lr"])


def build_avg_policy_stats(policy: Policy, batch) -> Dict[str, TensorType]:
    return {"loss": policy.loss}


class ManualLearningRateSchedule:
    """Mixin for TFPolicy that adds a learning rate schedule."""

    def __init__(self, lr, lr_schedule):
        self.cur_lr = lr
        if lr_schedule is None:
            self.lr_schedule = ConstantSchedule(lr, framework=None)
        else:
            self.lr_schedule = PiecewiseSchedule(
                lr_schedule, outside_value=lr_schedule[-1][-1], framework=None)

    # not called automatically by any rllib logic, call this in your training script or a trainer callback
    def update_lr(self, timesteps_total):
        print(f"cur lr {self.cur_lr}")
        self.cur_lr = self.lr_schedule.value(timesteps_total)
        for opt in self._optimizers:
            for p in opt.param_groups:
                p["lr"] = self.cur_lr


def setup_mixins(policy, obs_space, action_space, config):
    ManualLearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def move_model_to_device(policy: Policy, obs_space: gym.spaces.Space,
                      action_space: gym.spaces.Space,
                      config: TrainerConfigDict) -> None:
    policy.avg_model.to(policy.device)


NFSPTorchAveragePolicy = build_torch_policy(
    name="NFSPAveragePolicy",
    extra_action_out_fn=behaviour_logits_fetches,
    loss_fn=build_supervised_learning_loss,
    get_default_config=lambda: grl.algos.nfsp_rllib.nfsp.DEFAULT_CONFIG,
    make_model_and_action_dist=build_avg_model_and_distribution,
    action_sampler_fn=action_sampler,
    before_init=setup_mixins,
    after_init=move_model_to_device,
    extra_learn_fetches_fn=lambda policy: {"sl_loss": policy.loss},
    optimizer_fn=sgd_optimizer,
    stats_fn=build_avg_policy_stats,
    mixins=[ManualLearningRateSchedule, SafeSetWeightsPolicyMixin],
    # action_distribution_fn=get_distribution_inputs_and_class,
)
