from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
import numpy as np

import pandas as pd
import torch
from omegaconf import OmegaConf
from toolz import compose
from tqdm import tqdm

from tmnre.benchmark.paths import benchmark_paths
from sbibm.utils.io import get_int_from_csv
from swyft.utils import tensor_to_array


def send_tensor_to_array_else_identity(item: str):
    from torch import tensor  # Keep this for the function
    if isinstance(item, str):
        if item.split("(")[0] == "tensor" and "nan" not in item:
            evald = eval(item)
            return tensor_to_array(evald)
        elif item.split("(")[0] == "tensor" and "nan" in item:
            return np.array(float("NaN"))
        else:
            return item
    else:
        return item


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

    basepaths = [p.parent for p in Path(basepath).expanduser().rglob("metrics.csv")]

    # import ipdb; ipdb.set_trace()
    for basepath in tqdm(basepaths):
        row = {}

        # Read hydra config
        path_cfg = basepath / benchmark_paths.run_yaml
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        try:
            row["task"] = cfg["task"]["name"]
        except:
            continue
        row["num_simulations"] = compose(int, get_int_from_csv)(
            basepath / benchmark_paths.num_simulations_simulator
        )
        row["num_constraining_simulations"] = compose(int, get_int_from_csv)(
            basepath / benchmark_paths.num_constraining_simulations
        )
        row["num_observation"] = cfg["task"]["num_observation"]
        row["algorithm"] = cfg["algorithm"]["name"] if cfg["algorithm"]["name"] != 'cnre' else 'tmnre'
        row["seed"] = cfg["seed"]
        row["dimension"] = compose(int, get_int_from_csv)(
            basepath / benchmark_paths.parameter_dimension
        )
        row["rounds"] = compose(int, get_int_from_csv)(
            basepath / benchmark_paths.num_elapsed_rounds
        )

        # Read metric results
        path_metrics = basepath / benchmark_paths.metrics
        metrics = pd.read_csv(path_metrics)
        metrics = metrics.transform(lambda col: col.map(send_tensor_to_array_else_identity))
        row.update(metrics.iloc[0].to_dict())

        # Path and folder
        row["path"] = str(basepath.absolute())
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
    df.to_csv(args.output)
