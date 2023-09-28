from collections import deque
from typing import Optional, List, Callable, Dict

import numpy as np
import ray
from ray.rllib import Policy
import pyspiel
from pyspiel import Policy as OpenSpielPolicy
from open_spiel.python.algorithms import expected_game_score

from grl.rl_apps.psro.poker_utils import openspiel_policy_from_nonlstm_rllib_policy

def _exp3_distribution_from_weights(weights: np.ndarray, gamma: float = 0.0) -> np.ndarray:
    weights_sum = np.sum(weights)
    probs = (1.0 - gamma) * (weights / weights_sum) + (gamma / len(weights))

    # can delete this, just checking math correctness
    probs_sanity_check = np.asarray([(1.0 - gamma) * (w / weights_sum) + (gamma / len(weights)) for w in weights], dtype=np.float64)
    assert np.allclose(probs, probs_sanity_check), f"probs: {probs}, probs_sanity_check {probs_sanity_check}"
    assert np.isclose(np.sum(probs), 1.0), probs
    assert np.isclose(np.sum(probs_sanity_check), 1.0), probs_sanity_check

    return probs


@ray.remote(num_cpus=0)
class Exp3Solver:
    # Implemented using the following page as a guide
    # https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

    def __init__(self, num_actions: int, gamma: float, action_reward_averaging_window: int = 100,
                 initial_action_probs=None):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma needs to be in (0, 1], got {gamma}")
        self._gamma = gamma

        if initial_action_probs is None:
            self._weights = np.ones(shape=num_actions, dtype=np.float64)
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self._weights = np.asarray(initial_action_probs, dtype=np.float64).copy()

        self._probs_all_time_sum = self.get_action_probs(gamma=0.0)

        self._action_reward_averaging_window = action_reward_averaging_window
        self._last_seen_rewards_for_each_action = [None] * len(self._weights)

    def set_gamma(self, gamma: float):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma needs to be in (0, 1], got {gamma}")
        self._gamma = gamma

    def sample_action(self, gamma=None) -> int:
        if gamma is not None and not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")
        if gamma is None:
            gamma = self._gamma

        return np.random.choice(a=list(range(len(self._weights))),
                                p=self.get_action_probs(gamma=gamma))

    def get_action_probs(self, gamma=None) -> np.ndarray:
        if gamma is not None and not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")
        if gamma is None:
            gamma = self._gamma

        return _exp3_distribution_from_weights(weights=self._weights, gamma=gamma)

    def get_avg_weights_action_probs(self, gamma=0.0):
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")

        avg_weights = self._probs_all_time_sum / sum(self._probs_all_time_sum)
        assert np.isclose(1.0, sum(avg_weights)), sum(avg_weights)

        return _exp3_distribution_from_weights(weights=avg_weights, gamma=gamma)

    def register_reward_for_action(self, action: int, reward: float):
        if self._last_seen_rewards_for_each_action[action] is None:
            self._last_seen_rewards_for_each_action[action] = deque(maxlen=self._action_reward_averaging_window)

        self._last_seen_rewards_for_each_action[action].appendleft(reward)

    def _get_cached_reward_for_action(self, action: int):
        reward_deque = self._last_seen_rewards_for_each_action[action]
        if reward_deque is None:
            return None
        return np.mean(reward_deque)

    def perform_n_updates(self, n: int):

        for _ in range(n):
            action_probs = self.get_action_probs()
            sampled_action = np.random.choice(a=list(range(len(self._weights))), p=action_probs)
            sampled_action_probability = action_probs[sampled_action]
            sampled_action_reward = self._get_cached_reward_for_action(sampled_action)
            if sampled_action_reward is None:
                print(f"Dropping update for action {sampled_action}, no reward data")
                continue

            estimated_reward = 1.0 * sampled_action_reward / sampled_action_probability
            self._weights[sampled_action] *= np.exp(
                estimated_reward * self._gamma / len(self._weights))  # important that we use estimated reward here!

            current_action_probs = self.get_action_probs(gamma=0.0)
            assert np.isclose(1.0, sum(current_action_probs)), sum(current_action_probs)
            self._probs_all_time_sum += current_action_probs



@ray.remote(num_cpus=0)
class Exp3ModelBasedSolver:
    # Implemented using the following page as a guide
    # https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/

    def __init__(self,
                 num_actions: int,
                 gamma: float,
                 acting_player: int,
                 game_version: str,
                 game_parameters: dict,
                 initial_action_probs=None):

        self._acting_player = acting_player
        self._game_version = game_version
        self._game_parameters = game_parameters

        if self._game_parameters is None:
            if game_version in ["kuhn_poker", "leduc_poker", "leduc_poker_10_card"]:
                self._game_parameters = {
                    "players": 2
                }
            else:
                self._game_parameters = {}
        self._game_parameters = {k: v for k, v in self._game_parameters.items()}
        openspiel_game = pyspiel.load_game(game_version, self._game_parameters)
        if game_version in ["oshi_zumo", "goofspiel"]:
            openspiel_game = pyspiel.convert_to_turn_based(openspiel_game)
        self._openspiel_game = openspiel_game


        if not 0 < gamma <= 1:
            raise ValueError(f"gamma needs to be in (0, 1], got {gamma}")
        self._gamma = gamma

        if initial_action_probs is None:
            self._weights = np.ones(shape=num_actions, dtype=np.float64)
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self._weights = np.asarray(initial_action_probs, dtype=np.float64).copy()

        self._probs_all_time_sum = self.get_action_probs(gamma=0.0)

        self._action_reward_cache = None
        self._opponent_openspiel_policy: Optional[OpenSpielPolicy] = None
        self._action_openspiel_policy_cache: Optional[Dict[int, OpenSpielPolicy]] = None

    def set_gamma(self, gamma: float):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma needs to be in (0, 1], got {gamma}")
        self._gamma = gamma

    def sample_action(self, gamma=None) -> int:
        if gamma is not None and not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")
        if gamma is None:
            gamma = self._gamma

        return np.random.choice(a=list(range(len(self._weights))),
                                p=self.get_action_probs(gamma=gamma))

    def get_action_probs(self, gamma=None) -> np.ndarray:
        if gamma is not None and not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")
        if gamma is None:
            gamma = self._gamma

        return _exp3_distribution_from_weights(weights=self._weights, gamma=gamma)

    def get_avg_weights_action_probs(self, gamma=0.0):
        if not 0 <= gamma <= 1:
            raise ValueError(f"gamma needs to be in [0, 1], got {gamma}")

        avg_weights = self._probs_all_time_sum / sum(self._probs_all_time_sum)
        assert np.isclose(1.0, sum(avg_weights)), sum(avg_weights)

        return _exp3_distribution_from_weights(weights=avg_weights, gamma=gamma)

    def _get_model_based_reward_for_action(self,
                                           action: int,
                                           rllib_policy: Policy,
                                           opponent_rllib_policy: Policy,
                                           set_rllib_weights_for_action_and_policy: Callable[[int, Policy], None]) -> float:

        if action in self._action_reward_cache:
            return self._action_reward_cache[action]

        if self._opponent_openspiel_policy is None:
            self._opponent_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(
                openspiel_game=self._openspiel_game,
                game_version=self._game_version,
                game_parameters=self._game_parameters,
                rllib_policy=opponent_rllib_policy
            )

        original_rllib_policy_weights = None
        if action in self._action_openspiel_policy_cache:
            action_openspiel_policy: OpenSpielPolicy = self._action_openspiel_policy_cache[action]
        else:
            original_rllib_policy_weights = rllib_policy.get_weights()
            set_rllib_weights_for_action_and_policy(action, rllib_policy)
            action_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(
                openspiel_game=self._openspiel_game,
                game_version=self._game_version,
                game_parameters=self._game_parameters,
                rllib_policy=rllib_policy
            )

        opponent_player = 1 - self._acting_player
        eval_openspiel_policies = [None, None]
        eval_openspiel_policies[self._acting_player] = action_openspiel_policy
        eval_openspiel_policies[opponent_player] = self._opponent_openspiel_policy
        assert not any([e is None for e in eval_openspiel_policies]), eval_openspiel_policies

        gt_reward_for_action = expected_game_score.policy_value(
            state=self._openspiel_game.new_initial_state(),
            policies=eval_openspiel_policies)[self._acting_player]

        self._action_reward_cache[action] = gt_reward_for_action

        if original_rllib_policy_weights:
            rllib_policy.set_weights(weights=original_rllib_policy_weights)

        return gt_reward_for_action

    def perform_n_updates(self,
                          n: int,
                          rllib_policy: Policy,
                          opponent_rllib_policy: Policy,
                          set_rllib_weights_for_action_and_policy: Callable[[int, Policy], None]):

        # reset caches related for calculating model-based reward against current opponent policy
        self._action_reward_cache = {}
        self._opponent_openspiel_policy = None
        self._action_openspiel_policy_cache = {}

        for _ in range(n):
            action_probs = self.get_action_probs()
            sampled_action = np.random.choice(a=list(range(len(self._weights))), p=action_probs)
            sampled_action_probability = action_probs[sampled_action]
            sampled_action_reward = self._get_model_based_reward_for_action(
                action=sampled_action,
                rllib_policy=rllib_policy,
                opponent_rllib_policy=opponent_rllib_policy,
                set_rllib_weights_for_action_and_policy=set_rllib_weights_for_action_and_policy
            )

            # if sampled_action_reward is None:
            #     print(f"Dropping update for action {sampled_action}, no reward data")
            #     continue

            estimated_reward = 1.0 * sampled_action_reward / sampled_action_probability
            self._weights[sampled_action] *= np.exp(
                estimated_reward * self._gamma / len(self._weights))  # important that we use estimated reward here!

            current_action_probs = self.get_action_probs(gamma=0.0) # TODO Kevin uses gamma, not 0.0
            assert np.isclose(1.0, sum(current_action_probs)), sum(current_action_probs)
            self._probs_all_time_sum += current_action_probs

@ray.remote(num_cpus=0)
class RegretMatchingSolver:

    def __init__(self, num_actions: int, initial_action_probs=None):
        self.num_actions = num_actions
        self.strategy_sum = np.zeros(shape=num_actions, dtype=np.float64)
        self.action_reward_sums = np.zeros(shape=num_actions, dtype=np.float64)
        self.action_counts = np.zeros(shape=num_actions, dtype=np.int64)

        if initial_action_probs is None:
            self._strategy = np.ones(shape=num_actions, dtype=np.float64) / num_actions
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self._strategy = np.asarray(initial_action_probs, dtype=np.float64).copy()

        assert np.isclose(sum(self._strategy), 1.0)

        self.total_accumulated_reward = 0.0
        self.total_actions_taken = 0

        self._avg_strategy = self._strategy.copy()

    def sample_action(self, use_avg=False, use_exploration=False) -> int:
        strategy = self._avg_strategy if use_avg else self._strategy

        if use_exploration:
            eps = 0.05
            strategy = (1.0 - eps) * strategy + eps * (np.ones_like(strategy) / len(strategy))

        assert np.isclose(sum(strategy), 1.0)

        return np.random.choice(a=list(range(len(strategy))), p=strategy)

    def get_action_probs(self, use_avg=False) -> np.ndarray:
        strategy = self._avg_strategy if use_avg else self._strategy
        assert np.isclose(sum(strategy), 1.0)
        return strategy.copy()

    def register_reward_for_regret_sampled_action(self, action: int, reward: float):
        self.action_reward_sums[action] += reward
        self.action_counts[action] += 1

        self.total_accumulated_reward += reward
        self.total_actions_taken += 1

        if not np.any(self.action_counts == 0):
            self._update_strategy()

    def _update_strategy(self):
        avg_reward = self.total_accumulated_reward / self.total_actions_taken
        action_regrets = []
        for action in range(len(self.action_reward_sums)):
            avg_action_reward = self.action_reward_sums[action] / self.action_counts[action]
            action_regret = avg_action_reward - avg_reward
            action_regrets.append(action_regret)
        action_regrets = np.asarray(action_regrets, dtype=np.float64)

        assert len(action_regrets) == self.num_actions

        self._strategy = action_regrets / np.sum(action_regrets)

        self._strategy[self._strategy < 0] = 1e-8  # reset negative regrets to (near) zero
        summation = np.sum(self._strategy)
        if summation > 0:
            # normalise
            self._strategy /= summation
        else:
            # uniform distribution to reduce exploitability
            self._strategy = np.ones_like(self._strategy) / len(self._strategy)

        assert np.isclose(sum(self._strategy), 1.0)

        self.strategy_sum += self._strategy
        assert len(self.strategy_sum) == self.num_actions

        self._avg_strategy = self.strategy_sum / np.sum(self.strategy_sum)

        assert np.isclose(sum(self._avg_strategy), 1.0)


@ray.remote(num_cpus=0)
class RMBufferSolver:
    # https://medium.com/hackernoon/artificial-intelligence-poker-and-regret-part-1-36c78d955720

    def __init__(self, num_actions: int, action_reward_averaging_window: int = 100, initial_action_probs=None):
        self.num_actions = num_actions

        if initial_action_probs is None:
            self.strategy = np.ones(shape=num_actions, dtype=np.float64) / num_actions
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self.strategy = np.asarray(initial_action_probs, dtype=np.float64).copy()
        assert np.isclose(sum(self.strategy), 1.0)
        self.avg_strategy = self.strategy.copy()

        self.strategy_sum = np.zeros_like(self.strategy)
        self.regret_sum = np.zeros_like(self.strategy)

        self._action_reward_averaging_window = action_reward_averaging_window
        self._last_seen_rewards_for_each_action = [None] * len(self.strategy)

    def sample_action(self, use_avg=False, use_exploration=False) -> int:
        strategy = self.avg_strategy if use_avg else self.strategy

        if use_exploration:
            eps = 0.1
            strategy = (1.0 - eps) * strategy + eps * (np.ones_like(strategy) / len(strategy))

        assert np.isclose(sum(strategy), 1.0)

        return np.random.choice(a=list(range(len(strategy))), p=strategy)

    def get_action_probs(self, use_avg=False) -> np.ndarray:
        strategy = self.avg_strategy if use_avg else self.strategy
        assert np.isclose(sum(strategy), 1.0)
        return strategy.copy()

    def register_reward_for_action(self, action: int, reward: float):
        if self._last_seen_rewards_for_each_action[action] is None:
            self._last_seen_rewards_for_each_action[action] = deque(maxlen=self._action_reward_averaging_window)

        self._last_seen_rewards_for_each_action[action].appendleft(reward)

    def _get_cached_reward_for_action(self, action: int):
        reward_deque = self._last_seen_rewards_for_each_action[action]
        if reward_deque is None:
            return None
        return np.mean(reward_deque)

    def perform_n_updates(self, n: int):

        if any(e is None for e in self._last_seen_rewards_for_each_action):
            print("not updating RM because we dont have reward data for all actions")
            return

        all_action_rewards = np.asarray([self._get_cached_reward_for_action(a) for a in range(len(self.strategy))],
                                        dtype=np.float64)

        for _ in range(n):
            action_probs = self.get_action_probs()
            # sampled_action = np.random.choice(a=list(range(len(self.strategy))), p=action_probs)
            # sampled_action_probability = action_probs[sampled_action]
            # sampled_action_reward = self._get_cached_reward_for_action(sampled_action)

            action_regrets = all_action_rewards - (all_action_rewards @ action_probs.T)

            self.regret_sum += action_regrets

            assert len(self.regret_sum) == self.num_actions
            assert len(action_regrets) == self.num_actions

            self.strategy = np.copy(self.regret_sum)
            self.strategy[self.strategy < 0] = 1e-8  # reset negative regrets to (near) zero

            summation = sum(self.strategy)
            if summation > 0:
                # normalise
                self.strategy /= summation
            else:
                # uniform distribution to reduce exploitability
                self.strategy = np.repeat(1 / self.num_actions, self.num_actions)

            self.strategy_sum += self.strategy

            summation = sum(self.strategy_sum)
            if summation > 0:
                self.avg_strategy = self.strategy_sum / summation
            else:
                self.avg_strategy = np.repeat(1 / self.num_actions, self.num_actions)

from ray.rllib.utils.timer import TimerStat

@ray.remote(num_cpus=0)
class MWUSolver:
    # https://arxiv.org/pdf/2110.02134.pdf

    def __init__(self,
                 num_actions: int,
                 learning_rate: float,
                 action_reward_averaging_window: int = 1000,
                 initial_action_probs=None):

        self._num_actions = num_actions
        self._learning_rate = learning_rate

        if initial_action_probs is None:
            self._action_probs_t = np.ones(shape=num_actions, dtype=np.float64) / num_actions
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self._action_probs_t = np.asarray(initial_action_probs, dtype=np.float64).copy()
        assert np.isclose(1.0, sum(self._action_probs_t)), sum(self._action_probs_t)

        self._probs_all_time_sum = self.get_action_probs()

        self._action_reward_averaging_window = action_reward_averaging_window
        self._last_seen_rewards_for_each_action: List[Optional[deque]] = [None] * num_actions

        self.update_timer = TimerStat(window_size=100)
        self.add_action_timer = TimerStat(window_size=100)

    def sample_action(self) -> int:

        return np.random.choice(a=list(range(len(self._action_probs_t))),
                                p=self.get_action_probs())

    def get_action_probs(self) -> np.ndarray:
        return self._action_probs_t.copy()

    def get_avg_weights_action_probs(self) -> np.ndarray:
        avg_weights = self._probs_all_time_sum / sum(self._probs_all_time_sum)
        assert np.isclose(1.0, sum(avg_weights)), sum(avg_weights)
        return avg_weights

    def register_reward_for_action(self, action: int, reward: float):
        with self.add_action_timer:
            if self._last_seen_rewards_for_each_action[action] is None:
                self._last_seen_rewards_for_each_action[action] = deque(maxlen=self._action_reward_averaging_window)
            self._last_seen_rewards_for_each_action[action].appendleft(reward)

    def _get_cached_reward_for_action(self, action: int) -> Optional[float]:
        reward_deque = self._last_seen_rewards_for_each_action[action]
        if reward_deque is None:
            return None
        return np.mean(reward_deque)

    def perform_update(self, callback=None):
        with self.update_timer:
            action_probs_t_minus_1 = self.get_action_probs()

            for a, action_prob_t_minus_1 in enumerate(action_probs_t_minus_1):
                action_payoff_vs_opponent_t_minus_1 = self._get_cached_reward_for_action(a)
                if action_payoff_vs_opponent_t_minus_1 is None:
                    print(f"Dropping update for action {a}, no reward data")
                    continue

                self._action_probs_t[a] = action_prob_t_minus_1 * np.exp(
                    self._learning_rate * action_payoff_vs_opponent_t_minus_1)
            self._action_probs_t = self._action_probs_t / np.sum(self._action_probs_t)

            assert np.isclose(1.0, sum(self._action_probs_t)), sum(self._action_probs_t)
            self._probs_all_time_sum += self._action_probs_t

        if callback is not None:
            callback(self)

    def apply_fn(self, fn):
        fn(self)

    def reset_average_policy(self):
        self._probs_all_time_sum = self.get_action_probs()



@ray.remote(num_cpus=0)
class MWUModelBasedSolver:
    # https://arxiv.org/pdf/2110.02134.pdf

    def __init__(self,
                 num_actions: int,
                 learning_rate: float,
                 acting_player: int,
                 game_version: str,
                 game_parameters: dict,
                 initial_action_probs=None):

        self._num_actions = num_actions
        self._learning_rate = learning_rate

        self._acting_player = acting_player
        self._game_version = game_version
        self._game_parameters = game_parameters

        if self._game_parameters is None:
            if game_version in ["kuhn_poker", "leduc_poker", "leduc_poker_10_card"]:
                self._game_parameters = {
                    "players": 2
                }
            else:
                self._game_parameters = {}
        self._game_parameters = {k: v for k, v in self._game_parameters.items()}
        openspiel_game = pyspiel.load_game(game_version, self._game_parameters)
        if game_version in ["oshi_zumo", "goofspiel"]:
            openspiel_game = pyspiel.convert_to_turn_based(openspiel_game)
        self._openspiel_game = openspiel_game

        if initial_action_probs is None:
            self._action_probs_t = np.ones(shape=num_actions, dtype=np.float64) / num_actions
        else:
            assert len(
                initial_action_probs) == num_actions, f"len(initial_action_probs): {len(initial_action_probs)}, num_actions: {num_actions}"
            self._action_probs_t = np.asarray(initial_action_probs, dtype=np.float64).copy()
        assert np.isclose(1.0, sum(self._action_probs_t)), sum(self._action_probs_t)

        self._probs_all_time_sum = self.get_action_probs()

        self._action_reward_cache = None
        self._opponent_openspiel_policy: Optional[OpenSpielPolicy] = None
        self._action_openspiel_policy_cache: Optional[Dict[int, OpenSpielPolicy]] = None

    def sample_action(self) -> int:
        return np.random.choice(a=list(range(len(self._action_probs_t))),
                                p=self.get_action_probs())

    def get_action_probs(self) -> np.ndarray:
        return self._action_probs_t.copy()

    def get_avg_weights_action_probs(self) -> np.ndarray:
        avg_weights = self._probs_all_time_sum / sum(self._probs_all_time_sum)
        assert np.isclose(1.0, sum(avg_weights)), sum(avg_weights)
        return avg_weights

    def _get_model_based_reward_for_action(self,
                                           action: int,
                                           rllib_policy: Policy,
                                           opponent_rllib_policy: Policy,
                                           set_rllib_weights_for_action_and_policy: Callable[[int, Policy], None]) -> float:

        if action in self._action_reward_cache:
            return self._action_reward_cache[action]

        if self._opponent_openspiel_policy is None:
            self._opponent_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(
                openspiel_game=self._openspiel_game,
                game_version=self._game_version,
                game_parameters=self._game_parameters,
                rllib_policy=opponent_rllib_policy
            )

        original_rllib_policy_weights = None
        if action in self._action_openspiel_policy_cache:
            action_openspiel_policy: OpenSpielPolicy = self._action_openspiel_policy_cache[action]
        else:
            original_rllib_policy_weights = rllib_policy.get_weights()
            set_rllib_weights_for_action_and_policy(action, rllib_policy)
            action_openspiel_policy = openspiel_policy_from_nonlstm_rllib_policy(
                openspiel_game=self._openspiel_game,
                game_version=self._game_version,
                game_parameters=self._game_parameters,
                rllib_policy=rllib_policy
            )

        opponent_player = 1 - self._acting_player
        eval_openspiel_policies = [None, None]
        eval_openspiel_policies[self._acting_player] = action_openspiel_policy
        eval_openspiel_policies[opponent_player] = self._opponent_openspiel_policy
        assert not any([e is None for e in eval_openspiel_policies]), eval_openspiel_policies

        gt_reward_for_action = expected_game_score.policy_value(
            state=self._openspiel_game.new_initial_state(),
            policies=eval_openspiel_policies)[self._acting_player]

        self._action_reward_cache[action] = gt_reward_for_action

        if original_rllib_policy_weights:
            rllib_policy.set_weights(weights=original_rllib_policy_weights)

        return gt_reward_for_action


    def perform_update(self,
                       rllib_policy: Policy,
                       opponent_rllib_policy: Policy,
                       set_rllib_weights_for_action_and_policy: Callable[[int, Policy], None],
                       callback=None):

        # reset caches related for calculating model-based reward against current opponent policy
        self._action_reward_cache = {}
        self._opponent_openspiel_policy = None
        self._action_openspiel_policy_cache = {}

        action_probs_t_minus_1 = self.get_action_probs()

        for a, action_prob_t_minus_1 in enumerate(action_probs_t_minus_1):
            action_payoff_vs_opponent_t_minus_1 = self._get_model_based_reward_for_action(
                action=a,
                rllib_policy=rllib_policy,
                opponent_rllib_policy=opponent_rllib_policy,
                set_rllib_weights_for_action_and_policy=set_rllib_weights_for_action_and_policy
            )
            # if action_payoff_vs_opponent_t_minus_1 is None:
            #     print(f"Dropping update for action {a}, no reward data")
            #     continue

            self._action_probs_t[a] = action_prob_t_minus_1 * np.exp(
                self._learning_rate * action_payoff_vs_opponent_t_minus_1)
        self._action_probs_t = self._action_probs_t / np.sum(self._action_probs_t)

        assert np.isclose(1.0, sum(self._action_probs_t)), sum(self._action_probs_t)
        self._probs_all_time_sum += self._action_probs_t

        if callback is not None:
            callback(self)

    def apply_fn(self, fn):
        fn(self)

    def reset_average_policy(self):
        self._probs_all_time_sum = self.get_action_probs()
