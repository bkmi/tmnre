from argparse import ArgumentParser
from pathlib import Path
import yaml

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from tmnre.io import load_yaml


def compile_df(
    basepath: str,
) -> pd.DataFrame:
    """Compile dataframe for further analyses

    `basepath` is the path to a folder over which to recursively loop. All information
    is compiled into a big dataframe and returned for further analyses.

    Args:
        basepath: Base path to use

    Returns:
        Dataframe with results
    """
    df = []

    basepaths = [p.parent for p in Path(basepath).expanduser().rglob("marginals.yaml")]

    for path_base in tqdm(basepaths):
        row = {}

        # Read hydra config
        path_cfg = path_base / "run.yaml"
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        try:
            row["task"] = cfg["task"]["name"]
        except:
            continue
        row["num_simulations"] = cfg["task"]["num_simulations"]
        row["num_observation"] = cfg["task"]["num_observation"]
        row["algorithm"] = cfg["algorithm"]["name"]
        row["seed"] = cfg["seed"]

        # Read marginal results
        path_marginals = path_base / "marginals.yaml"
        if path_marginals.exists():
            try:
                # uses old string of list key technique
                conf = OmegaConf.load(str(path_marginals))
                marginals = OmegaConf.to_container(conf)
            except yaml.constructor.ConstructorError:
                # uses new tuple key technique
                marginals = load_yaml(str(path_marginals))

            for marginal_name, marginal_value in marginals.items():
                if np.isfinite(marginal_value):
                    row[marginal_name] = marginal_value
                else:
                    continue
        else:
            continue

        # Read marginal stats
        path_stats = path_base / "stats.yaml"
        if path_stats.exists():
            stats = OmegaConf.to_container(OmegaConf.load(str(path_stats)))
            for stats_name, stats_value in stats.items():
                row[stats_name] = stats_value
        else:
            continue

        # Path and folder
        row["path"] = str((path_base).absolute())
        row["folder"] = row["path"].split(str(basepath) + "/")[-1]

        df.append(row)

    df = pd.DataFrame().from_dict(df)
    if len(df) > 0:
        df["num_observation"] = df["num_observation"].astype("category")

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    df = compile_df(args.directory)
    df = df.drop_duplicates(["task", "algorithm", "num_simulations", "num_observation"])
    df.to_csv(args.output)
    print(len(df))
    print(df.head())
