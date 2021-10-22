from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from toolz import compose
from tqdm import tqdm

from sbibm.utils.io import get_float_from_csv, get_int_from_csv, get_ndarray_from_csv
from tmnre.benchmark.paths import benchmark_paths


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

    basepaths = [p.parent for p in Path(basepath).expanduser().rglob("c2sts.csv")]

    for basepath in tqdm(basepaths):
        row = {}

        # Read hydra config
        path_cfg = basepath / ".hydra/config.yaml"
        if path_cfg.exists():
            cfg = OmegaConf.to_container(OmegaConf.load(str(path_cfg)))
        else:
            continue

        # Config file
        row["epsilon"] = cfg["epsilon"]
        row["num_simulations"] = get_ndarray_from_csv(
            basepath / "n-simulations.csv",
        ).flatten()[-1]
        row["volume"] = get_ndarray_from_csv(
            basepath / "volumes.csv",
        ).flatten()[-1]
        row["kld"] = get_ndarray_from_csv(
            basepath / "kld.csv",
        ).flatten()[-1]
        row["seed"] = get_int_from_csv(basepath / "seed.csv")

        c2st = pd.read_csv(basepath / "c2sts.csv")
        row.update(c2st.iloc[0].to_dict())

        # Path and folder
        row["path"] = str(basepath.absolute())
        row["folder"] = row["path"].split(str(basepath) + "/")[-1]

        df.append(row)

    df = pd.DataFrame().from_dict(df)

    return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    df = compile_df(args.directory)
    df.to_csv(args.output)
