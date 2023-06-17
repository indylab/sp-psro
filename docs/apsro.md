# Running APSRO Experiments

Instructions to reproduce neural experiments from [Anytime PSRO for Two-Player Zero-Sum Games](https://arxiv.org/abs/2201.07700)

## Launching Main Experiments

Prior to all main experiment scripts, run:
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

## Leduc Poker
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
python launch_psro_as_single_script.py  --scenario leduc_psro_dqn_regret --instant_first_iter
```


## Goofspiel
**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario goofspiel_psro_dqn
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario goofspiel_psro_dqn --instant_first_iter
```


**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py  --scenario goofspiel_psro_dqn --instant_first_iter
```


## 2D Continuous-Action Hill-Climbing Game
**APSRO** (run both scripts together, launch manager first)
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python general_psro_manager.py --scenario loss_game_psro_10_moves_alpha_2.7
```
```bash
conda activate grl_dev; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=; export GRL_SEED=$(tmux display-message -p '#I')
python anytime_psro_br_both_players.py.py --scenario loss_game_psro_10_moves_alpha_2.7 --instant_first_iter
```

**PSRO**
```bash
conda activate grl_dev; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=; export GRL_SEED="10$(tmux display-message -p '#I')00$(tmux display -pt "${TMUX_PANE:?}" '#{pane_index}')"
python launch_psro_as_single_script.py  --scenario loss_game_psro_10_moves_alpha_2.7 --instant_first_iter
```


# Measuring Approximate Exploitability for Large Games
TODO

# Graphing results 
See [notebooks](/notebooks) for example scripts to graph exploitability vs experience collected. 

For smaller games, exact exploitability is logged during training. For larger games like Goofspiel and the 2D Continuous Hill-Climbing Game, approximate exploitability needs to be separately estimated by training best-responses against checkpoints in a standalone script. 
