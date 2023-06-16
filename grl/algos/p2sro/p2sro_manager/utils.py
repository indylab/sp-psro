from typing import Dict, List, Union, Optional

import numpy as np

from grl.algos.p2sro.payoff_table import PayoffTable
from grl.utils.strategy_spec import StrategySpec


class PolicySpecDistribution(object):

    def __init__(self, payoff_table: PayoffTable, player: int,
                 policy_selection_probs_indexed_by_policy_num: List[float]):

        self._policy_specs_to_probs = {
            payoff_table.get_spec_for_player_and_pure_strat_index(
                player=player, pure_strat_index=policy_num): selection_prob
            for policy_num, selection_prob in enumerate(policy_selection_probs_indexed_by_policy_num)
        }

        self.player = player

    def sample_policy_spec(self) -> StrategySpec:
        return np.random.choice(a=list(self._policy_specs_to_probs.keys()),
                                p=list(self._policy_specs_to_probs.values()))

    def probabilities_for_each_strategy(self) -> np.ndarray:
        return np.asarray(list(self._policy_specs_to_probs.values()), dtype=np.float64)

    def spec_list(self) -> List[StrategySpec]:
        return list(self._policy_specs_to_probs.keys())

    def get_spec_by_index(self, index) -> StrategySpec:
        return list(self._policy_specs_to_probs.keys())[index]

    def __len__(self):
        return len(self._policy_specs_to_probs)


def get_latest_metanash_strategies(payoff_table: PayoffTable,
                                   as_player: int,
                                   as_policy_num: int,
                                   fictitious_play_iters: Optional[int] = None,
                                   mix_with_uniform_dist_coeff: float = 0.0,
                                   include_policies_for_each_player: List[List[int]] = None,
                                   print_matrix: bool = True,
                                   use_lp_solver: bool = False) -> Union[None, Dict[int, PolicySpecDistribution]]:
    # Currently this function only handles 2-player games
    if as_policy_num is None:
        as_policy_num = payoff_table.shape()[as_player]

    if not 0 <= as_player < payoff_table.n_players():
        raise ValueError(f"as_player {as_player} should be in the range [0, {payoff_table.n_players()}).")

    if include_policies_for_each_player is None:
        opponent_player = int(1 - as_player)
        if (payoff_table.shape() == (0,) and as_policy_num != 0) or \
                (payoff_table.shape() != (0,) and payoff_table.shape()[opponent_player] < as_policy_num):
            raise ValueError(f"In the payoff table, policy_num {as_policy_num} is out of range for player {as_player}. "
                             f"Payoff table shape is {payoff_table.shape()}.")

    if payoff_table.n_players() != 2:
        raise NotImplemented("Solving normal form Nash equilibrium strats for >2 player games not implemented.")

    if as_policy_num == 0:
        return None


    other_players = list(range(0, payoff_table.n_players()))
    other_players.remove(as_player)

    # check that all payoffs are zero-sum, otherwise ok if the game is symmetric
    if not np.array_equal(payoff_table.get_payoff_matrix_for_player(player=0),
                          payoff_table.get_payoff_matrix_for_player(player=1)):
        assert np.all(np.isclose(
            payoff_table.get_payoff_matrix_for_player(player=0),
            -1 * payoff_table.get_payoff_matrix_for_player(player=1)
        )), f"player 0 matrix:\n{payoff_table.get_payoff_matrix_for_player(player=0)}\n\n" \
            f"player 1 matrix:\n{payoff_table.get_payoff_matrix_for_player(player=1)}\n\n"

    opponent_strategy_distributions = {}
    for other_player in other_players:
        player_payoff_matrix = payoff_table.get_payoff_matrix_for_player(player=other_player)
        games_played_matrix = payoff_table.get_games_played_matrix_for_player(player=other_player)
        assert len(player_payoff_matrix.shape) == 2  # assume a 2D payoff matrix

        # print(f"original payoff matrix: {player_payoff_matrix}, include_policies_for_each_player: {include_policies_for_each_player}")

        if include_policies_for_each_player:
            for p in range(2):
                if len(include_policies_for_each_player[p]) == 0:
                    return None

            # check that these are sequential lists starting from 0
            assert np.array_equal(include_policies_for_each_player[0], list(range(0, len(include_policies_for_each_player[0])))), include_policies_for_each_player
            assert np.array_equal(include_policies_for_each_player[1], list(range(0, len(include_policies_for_each_player[1])))), include_policies_for_each_player

            player_payoff_matrix = player_payoff_matrix[include_policies_for_each_player[0], :]
            player_payoff_matrix = player_payoff_matrix[:, include_policies_for_each_player[1]]

            games_played_matrix = games_played_matrix[include_policies_for_each_player[0], :]
            games_played_matrix = games_played_matrix[:, include_policies_for_each_player[1]]

            # print(f"cropped payoff matrix: {player_payoff_matrix}")

        else:
            # only consider policies below 'as_policy_num' in the p2sro hierarchy
            player_payoff_matrix = player_payoff_matrix[:as_policy_num, :as_policy_num]
            games_played_matrix = games_played_matrix[:as_policy_num, :as_policy_num]

        # Every payoff value should be the result of a non-zero amount of games played unless there's only one strategy.
        # Having no games played for only diagonal entries is ok for symmetrics game
        # where the same population is used for both sides.
        assert 0 not in games_played_matrix or np.size(games_played_matrix) == 1 or \
               (np.all(np.diagonal(games_played_matrix) == 0) and
                0 not in np.extract(1 - np.eye(len(games_played_matrix)), games_played_matrix)), f"games played: \n{games_played_matrix}\npayoffs:\n{player_payoff_matrix}"


        player_payoff_matrix_current_player_is_rows = player_payoff_matrix.transpose((other_player, as_player))

        if print_matrix:
            print(f"payoff matrix as {other_player} (row) against {as_player} (columns):")
            print(player_payoff_matrix_current_player_is_rows)

        if use_lp_solver:
            from open_spiel.python.algorithms import lp_solver
            import pyspiel
            nash_prob_1, nash_prob_2, _, _ = lp_solver.solve_zero_sum_matrix_game(pyspiel.create_matrix_game(player_payoff_matrix_current_player_is_rows, -player_payoff_matrix_current_player_is_rows))
            selection_probs = np.copy(nash_prob_1).reshape((-1,))
            assert np.isclose(sum(selection_probs), 1.0), sum(selection_probs)
            for i, prob in enumerate(selection_probs):
                if prob < 0.0:
                    selection_probs[i] = 0.0
            selection_probs = selection_probs / sum(selection_probs)
            assert np.isclose(sum(selection_probs), 1.0), sum(selection_probs)
        else:
            row_averages, col_averages, exps = fictitious_play(iters=fictitious_play_iters,
                                                               payoffs=player_payoff_matrix_current_player_is_rows)
            selection_probs = np.copy(row_averages[-1])

        if mix_with_uniform_dist_coeff is not None and mix_with_uniform_dist_coeff > 0:
            uniform_dist = np.ones_like(selection_probs) / len(selection_probs)
            selection_probs = mix_with_uniform_dist_coeff * uniform_dist + (
                    1.0 - mix_with_uniform_dist_coeff) * selection_probs

        opponent_strategy_distributions[other_player] = PolicySpecDistribution(
            payoff_table=payoff_table, player=other_player,
            policy_selection_probs_indexed_by_policy_num=selection_probs)

    return opponent_strategy_distributions


def get_br_to_strat(strat, payoffs, strat_is_row=True, verbose=False):
    if strat_is_row:
        weighted_payouts = strat @ payoffs
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmin(weighted_payouts)] = 1
        idx = np.argmin(weighted_payouts)
    else:
        weighted_payouts = payoffs @ strat.T
        if verbose:
            print("strat ", strat)
            print("weighted payouts ", weighted_payouts)

        br = np.zeros_like(weighted_payouts)
        br[np.argmax(weighted_payouts)] = 1
        idx = np.argmax(weighted_payouts)
    return br, idx


def fictitious_play(payoffs, iters=2000, verbose=False):
    row_dim = payoffs.shape[0]
    col_dim = payoffs.shape[1]
    row_pop = np.random.uniform(0, 1, (1, row_dim))
    row_pop = row_pop / row_pop.sum(axis=1)[:, None]
    row_averages = row_pop
    col_pop = np.random.uniform(0, 1, (1, col_dim))
    col_pop = col_pop / col_pop.sum(axis=1)[:, None]
    col_averages = col_pop
    exps = []
    for i in range(iters):
        row_average = np.average(row_pop, axis=0)
        col_average = np.average(col_pop, axis=0)

        row_br, idx = get_br_to_strat(col_average, payoffs, strat_is_row=False, verbose=False)
        col_br, idx = get_br_to_strat(row_average, payoffs, strat_is_row=True, verbose=False)

        exp1 = row_average @ payoffs @ col_br.T
        exp2 = row_br @ payoffs @ col_average.T
        exps.append(exp2 - exp1)
        if verbose:
            print(exps[-1], "exploitability")

        row_averages = np.vstack((row_averages, row_average))
        col_averages = np.vstack((col_averages, col_average))

        row_pop = np.vstack((row_pop, row_br))
        col_pop = np.vstack((col_pop, col_br))
    return row_averages, col_averages, exps
