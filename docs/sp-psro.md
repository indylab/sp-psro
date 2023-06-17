# Running Self-Play PSRO Experiments

Instructions to reproduce neural experiments from [Self-Play PSRO: Toward Optimal Populations in
Two-Player Zero-Sum Games](https://arxiv.org/abs/2207.06541)

## Launching Main Experiments

Prior to all main experiment scripts:
```bash
conda activate grl_dev
export CUDA_VISIBLE_DEVICES=
```

Experiments with multiple concurrent scripts have the scripts find each other's network ports to communicate based on the scenario name and environment variable `GRL_SEED`.
If launching multiple scenarios on the same machine, launch each pair of scripts with a different value for `GRL_SEED`.

Also launch each `launch_psro_as_single_script.py` for a single scenario with a different value for `GRL_SEED` since it launches multiple scripts under the hood.

If using tmux (recommended), the following commands are helpful:

Set `GRL_SEED` to tmux window number (basically a tab in tmux). Useful for syncing multiple scripts in different panes in the same window, for example launching all scripts for a single APSRO run in the same window:
```bash
export GRL_SEED=$(tmux display-message -p '#I')
```

Set `GRL_SEED` to tmux window+pane number. Useful for differentiating multiple scripts in different panes in the same window, for example launching x3 independent PSRO runs in the same window:
```bash
export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
```

## Liar's Dice
**SP-PSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario liars_dice_psro_dqn_shorter_avg_pol
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_avg_policy_br_both_players.py.py --scenario liars_dice_psro_dqn_shorter_avg_pol --instant_first_iter --avg_policy_learning_rate 0.1 --train_avg_policy_for_n_iters_after 10000 --force_sp_br_play_rate 0.05
```

**SP-PSRO Last-Iterate Ablation** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario liars_dice_psro_dqn_shorter
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_last_iterate_br_both_players.py.py --scenario liars_dice_psro_dqn_shorter --instant_first_iter --force_sp_br_play_rate 0.05
```

**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario liars_dice_psro_dqn_shorter
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario liars_dice_psro_dqn_shorter --instant_first_iter
```



**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py --instant_first_iter --scenario liars_dice_psro_dqn_shorter
```


## Small Battleship

**SP-PSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario battleship_psro_dqn_avg_pol
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_avg_policy_br_both_players.py.py --scenario battleship_psro_dqn_avg_pol --instant_first_iter --avg_policy_learning_rate 0.1 --train_avg_policy_for_n_iters_after 10000 --force_sp_br_play_rate 0.1
```

**SP-PSRO Last-Iterate Ablation** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario battleship_psro_dqn
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_last_iterate_br_both_players.py.py --scenario battleship_psro_dqn --instant_first_iter --force_sp_br_play_rate 0.1
```

**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario battleship_psro_dqn
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario battleship_psro_dqn --instant_first_iter
```


**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py --instant_first_iter --scenario battleship_psro_dqn
```


## 4-Repeated Rock-Paper-Scissors

**SP-PSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario repeated_rps_psro_dqn_avg_pol
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_avg_policy_br_both_players.py.py --scenario repeated_rps_psro_dqn_avg_pol --instant_first_iter --avg_policy_learning_rate 0.1 --train_avg_policy_for_n_iters_after 10000 --force_sp_br_play_rate 0.1
```

**SP-PSRO Last-Iterate Ablation** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario repeated_rps_psro_dqn
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_last_iterate_br_both_players.py.py --scenario repeated_rps_psro_dqn --instant_first_iter --force_sp_br_play_rate 0.1
```

**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario repeated_rps_psro_dqn
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario repeated_rps_psro_dqn --instant_first_iter
```

**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py --instant_first_iter --scenario repeated_rps_psro_dqn
```

## Leduc Poker

**SP-PSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario leduc_psro_dqn_regret_avg_pol
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_avg_policy_br_both_players.py.py --scenario leduc_psro_dqn_regret_avg_pol --instant_first_iter --avg_policy_learning_rate 0.1 --train_avg_policy_for_n_iters_after 10000 --force_sp_br_play_rate 0.1
```

**SP-PSRO Last-Iterate Ablation** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario leduc_psro_dqn_regret
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python self_play_psro_last_iterate_br_both_players.py.py --scenario leduc_psro_dqn_regret --instant_first_iter --force_sp_br_play_rate 0.1
```

**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario leduc_psro_dqn_regret
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario leduc_psro_dqn_regret --instant_first_iter
```

**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py --instant_first_iter --scenario leduc_psro_dqn_regret
```

# Graphing results 
See [notebooks](/notebooks) for example scripts to graph exploitability vs experience collected. 


