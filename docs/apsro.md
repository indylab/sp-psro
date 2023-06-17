# Running APSRO Experiments

Instructions to reproduce neural experiments from [Anytime PSRO for Two-Player Zero-Sum Games](https://arxiv.org/abs/2201.07700)

First see [installation](/docs/install.md) documentation for setting up the dev environment.


## Launching Main Experiments

Experiment details and hyperparameters are organized in uniquely named `Scenarios`. When launching a learning script, you will generally specify a scenario name as a command-line argument. Experiment scenarios are defined in [grl/rl_apps/scenarios/catalog](/grl/rl_apps/scenarios/catalog).

Our PSRO/APSRO implementation consists of multiple scripts that are launched on separate terminals:
- The manager script (to track the population and, for PSRO, track the payoff table and launch empirical payoff evaluations)
- Scripts to run RL best response learners for each of the 2 players

The manager acts as a server that the best response learners connect to via gRPC.

([tmux](https://github.com/tmux/tmux/wiki) with a [nice configuration](https://github.com/gpakosz/.tmux) is useful for managing and organizing many terminal sessions)

### To launch a PSRO Experiment
```shell
# from the repository root
cd grl/rl_apps/psro
python general_psro_manager.py --scenario <my_scenario_name>
```
```shell
# in a 2nd terminal
cd grl/rl_apps/psro
python general_psro_br.py --player 0 --scenario <same_scenario_as_manager> --instant_first_iter
```
```shell
# in a 3rd terminal
cd grl/rl_apps/psro
python general_psro_br.py --player 1 --scenario <same_scenario_as_manager> --instant_first_iter
``` 

### To launch an APSRO Experiment
```shell
# from the repository root
cd grl/rl_apps/psro
python general_psro_manager.py --scenario <my_scenario_name>
```
```shell
# in a 2nd terminal
cd grl/rl_apps/psro
python anytime_psro_br_both_players.py --scenario <my_scenario_name> --instant_first_iter
```



If launching each of these scripts on the same computer, the best response scripts will automatically connect to a manager running the same scenario/seed  on a randomized port defined by the manager in `\tmp\grl_ports.json`. Otherwise, pass the `--help` argument to these scripts to see options for specifying hosts and ports. 

Multiple experiments with the same scenario can be launched on a single host by setting the `GRL_SEED` environment variable to a different integer value for each set of corresponding experiments. If unset, `GRL_SEED` defaults to 0. Best response processes will automatically connect to a manager server with the same scenario and `GRL_SEED`.


## Leduc Poker
**APSRO** (run both scripts together, launch manager first)
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python general_psro_manager.py --scenario leduc_psro_dqn_regret
```
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python anytime_psro_br_both_players.py --scenario leduc_psro_dqn_regret --instant_first_iter
```



**PSRO**
```bash
conda activate sp_psro; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=
python launch_psro_as_single_script.py  --scenario leduc_psro_dqn_regret --instant_first_iter
```


## Goofspiel
**APSRO** (run both scripts together, launch manager first)
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python general_psro_manager.py --scenario goofspiel_psro_dqn
```
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python anytime_psro_br_both_players.py --scenario goofspiel_psro_dqn --instant_first_iter
```


**PSRO**
```bash
conda activate sp_psro; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=
python launch_psro_as_single_script.py  --scenario goofspiel_psro_dqn --instant_first_iter
```


## 2D Continuous-Action Hill-Climbing Game
**APSRO** (run both scripts together, launch manager first)
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python general_psro_manager.py --scenario loss_game_psro_10_moves_alpha_2.7
```
```bash
conda activate sp_psro; cd ~/git/grl/grl/rl_apps/psro; export CUDA_VISIBLE_DEVICES=
python anytime_psro_br_both_players.py --scenario loss_game_psro_10_moves_alpha_2.7 --instant_first_iter
```

**PSRO**
```bash
conda activate sp_psro; cd ~/git/grl/examples; export CUDA_VISIBLE_DEVICES=
python launch_psro_as_single_script.py  --scenario loss_game_psro_10_moves_alpha_2.7 --instant_first_iter
```


# Measuring Approximate Exploitability for Large Games
TODO add examples for this

# Graphing results 
See [notebooks](/notebooks) for example scripts to graph exploitability vs experience collected. 

For smaller games, exact exploitability is logged during training. For larger games like Goofspiel and the 2D Continuous Hill-Climbing Game, approximate exploitability needs to be separately estimated by training best-responses against checkpoints in a standalone script. 
