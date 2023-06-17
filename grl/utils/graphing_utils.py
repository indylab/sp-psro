import itertools
import json
import os
from typing import Callable, Optional, Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_run_results(
        non_psro_run_dict: Dict[str, str],
        psro_run_dict: Dict[str, str],
        resolve_missing_file_before_read_callback: Optional[Callable[[str], str]] = None,
        parse_approximate_exploitability: bool = False,
        manually_add_initial_untrained_policy_exploitability: Optional[float] = None) -> pd.DataFrame:

    runs_to_parsed_results = {}
    for name, json_path in non_psro_run_dict.items():
        runs_to_parsed_results[name] = {}
        timesteps = []
        episodes = []
        exploitability = []
        print(f"parsing {json_path}")

        if resolve_missing_file_before_read_callback is not None:
            json_path = resolve_missing_file_before_read_callback(json_path)

        with open(json_path, "r") as json_file:
            for line in json_file:
                try:
                    json_result = json.loads(s=line)
                except json.JSONDecodeError:
                    break

                timesteps_entry = json_result["timesteps_total"]
                episodes_entry = json_result["episodes_total"]
                try:
                    exploitability_entry = (json_result.get("avg_policy_exploitability")
                                            or json_result.get("exploitability")
                                            or json_result["approx_exploitability"])
                except KeyError:
                    continue

                timesteps.append(timesteps_entry)
                episodes.append(episodes_entry)
                exploitability.append(exploitability_entry)

        runs_to_parsed_results[name]["timesteps"] = timesteps
        runs_to_parsed_results[name]["episodes"] = episodes
        runs_to_parsed_results[name]["exploitability"] = exploitability

    timesteps = list(itertools.chain(*[v["timesteps"] for k, v in runs_to_parsed_results.items()]))
    episodes = list(itertools.chain(*[v["episodes"] for k, v in runs_to_parsed_results.items()]))
    exploitability = list(itertools.chain(*[v["exploitability"] for k, v in runs_to_parsed_results.items()]))
    run = list(itertools.chain(*[[k[:-2]] * len(v["exploitability"]) for k, v in runs_to_parsed_results.items()]))

    df_in_dict = {
        "timesteps": timesteps,
        "episodes": episodes,
        "exploitability": exploitability,
        "run": run,
    }

    for run_name, data_path in psro_run_dict.items():
        try:
            if not os.path.exists(data_path):
                data_path = resolve_missing_file_before_read_callback(data_path)
            with open(data_path, "r") as json_file:
                data = json.load(json_file)

            apsro_spsro_algorithm_names = ["ao-psro", "apsro", "s-psro", "spsro", "sp-psro"]

            if parse_approximate_exploitability:
                exploitability_key = "approx_exploitability"
                if any(algorithm_name in run_name.lower() for algorithm_name in apsro_spsro_algorithm_names):
                    exploitability_key = "exploitability"
                    if "approx" not in os.path.basename(data_path):
                        raise ValueError(f"Expecting apsro or sp-psro states file to have 'approx' in the name "
                                         f"to indicate it's logging approximate exploitability. "
                                         f"Got {os.path.basename(data_path)}")
            else:
                exploitability_key = "exploitability"
                if any(algorithm_name in run_name.lower() for algorithm_name in apsro_spsro_algorithm_names):
                    exploitability_key = "alternate_metasolver_exploitability"

            filtered_data = {}
            for key, column in data.items():
                filtered_data[key] = []
                for i, item in enumerate(column):

                    try:
                        if data[exploitability_key][i] is not None and data[exploitability_key][i] != -1:
                            filtered_data[key].append(data[key][i])
                    except KeyError:
                        if any(algorithm_name in run_name.lower() for algorithm_name in apsro_spsro_algorithm_names):
                            continue

            data = filtered_data

            timesteps = data.get("timesteps") or data.get("timesteps_total") or data["total_steps"]
            episodes = data.get("episodes") or data.get("episodes_total") or data["total_episodes"]
            exploitability = data.get(exploitability_key) or data["approx_exploitability"]
            assert len(timesteps) == len(episodes)
            assert len(exploitability) == len(timesteps)
            df_in_dict["timesteps"].extend(timesteps)
            df_in_dict["episodes"].extend(episodes)
            df_in_dict["exploitability"].extend(exploitability)
            df_in_dict["run"].extend(run_name[:-2] for _ in exploitability)
        except:
            print(f"error occurred with run: {run_name}")
            raise

        if manually_add_initial_untrained_policy_exploitability is not None:
            # add random policy initial exploitability
            if 0 not in episodes:
                df_in_dict["timesteps"].append(0)
                df_in_dict["episodes"].append(0)
                df_in_dict["exploitability"].append(manually_add_initial_untrained_policy_exploitability)
                df_in_dict["run"].append(run_name[:-2])

    return pd.DataFrame.from_dict(data=df_in_dict)


def graph_df_and_save_figure(
        non_psro_run_dict: Dict[str, str],
        psro_run_dict: Dict[str, str],
        df: pd.DataFrame,
        title: str,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        use_approx_exploitability: bool = False,
):

    font = {'family': 'normal',
            'weight': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)
    from matplotlib import rcParams
    rcParams.update({
        'figure.autolayout': True,
        # "figure.figsize": (16*1.5, 9*1.5)
    })

    individual_experiment_dfs = {}
    for run_dict in [psro_run_dict, non_psro_run_dict]:
        for key in run_dict.keys():
            name = key[:-2]
            if name not in individual_experiment_dfs and all(f"{name} {i}" in run_dict.keys() for i in [1, 2, 3]):
                # 3 seeds
                _df_experiment = df.loc[df['run'] == name]
                _df_experiment["quantized_episodes"] = _df_experiment["episodes"]
                _df_experiment["quantized_episodes"] = [interval.mid for interval in pd.qcut(_df_experiment["episodes"],
                                                                                             len(_df_experiment) // 3).tolist()]
                individual_experiment_dfs[name] = _df_experiment
            elif name not in individual_experiment_dfs:
                # 1 seed
                _df_experiment = df.loc[df['run'] == name]
                _df_experiment["quantized_episodes"] = _df_experiment["episodes"]
                individual_experiment_dfs[name] = _df_experiment

    for run_label, run_df in individual_experiment_dfs.items():
        fig = sns.lineplot(y="exploitability", x="quantized_episodes", data=run_df, err_style="band", legend="brief",
                           label=run_label)

    plt.title(title)
    plt.legend(frameon=False, title=None, loc="upper right")
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    x1, x2, y1, y2 = plt.axis()
    print((x1, x2, y1, y2))
    plt.xlabel("Episodes")

    if use_approx_exploitability:
        plt.ylabel("Approx. Exploitability")
    else:
        plt.ylabel("Exploitability")

    save_file_path = f'{title}.png'.replace("'", "").replace(" ", "_")
    print(f"saving to {save_file_path}")
    plt.savefig(save_file_path)
