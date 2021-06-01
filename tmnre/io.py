from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

from sbibm.utils.io import get_ndarray_from_csv, save_tensor_to_csv
from swyft.types import Array
from swyft.utils.array import tensor_to_array


def save_yaml(file, dictionary):
    with open(file, "w") as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False)


def load_yaml(file):
    with open(file, "r") as f:
        dictionary = yaml.load(f)
    return dictionary


def save_config(cfg: DictConfig, filename: str = "run.yaml") -> None:
    """Saves config as yaml

    Args:
        cfg: Config to store
        filename: Filename
    """
    with open(filename, "w") as fh:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True), fh, default_flow_style=False
        )


def save_marginal_samples_to_csv_directory(
    root: Union[str, Path],
    samples: Dict[str, np.ndarray],
    dtype: type = np.float32,
    index: bool = False,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for param_tuple, marginal_samples in samples.items():
        param_tuple = [str(t) for t in param_tuple]
        path = Path(root, "-".join(param_tuple) + ".csv.bz2")
        save_ndarray_to_csv(
            path=path.absolute(),
            data=marginal_samples,
            columns=param_tuple,
            dtype=dtype,
            index=index,
        )


def load_marginal_samples_from_csv_directory(
    root: Union[str, Path], dtype: type = np.float32, atleast_2d: bool = True
) -> Dict[str, np.ndarray]:
    path = Path(root)
    csvs = path.glob("*.csv*")

    out = {}
    for csv in csvs:
        df = pd.read_csv(csv)
        key = tuple(int(c) for c in df.columns)
        out[key] = get_ndarray_from_csv(csv, dtype, atleast_2d)
    return out


def save_ndarray_to_csv(
    path: Union[str, Path],
    data: np.ndarray,
    columns: Optional[Iterable[str]] = None,
    dtype: type = np.float32,
    index: bool = False,
):
    pd.DataFrame(data.astype(dtype), columns=columns).to_csv(path, index=index)


def save_array_to_csv(
    path: Union[str, Path],
    data: Array,
    columns: Optional[Iterable[str]] = None,
    dtype: type = np.float32,
    index: bool = False,
):
    if isinstance(data, torch.Tensor):
        save_tensor_to_csv(path, data, columns, dtype, index)
    else:
        data = np.asarray(data)
        save_ndarray_to_csv(path, data, columns, dtype, index)


def save_weighted_samples_to_csv_directory(
    root: Union[str, Path],
    weighted_samples: Dict[str, dict],
    dtype: type = np.float32,
    index: bool = False,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    params_path = Path(root, "params.csv.bz2")
    save_array_to_csv(params_path, weighted_samples["params"], dtype=dtype, index=index)
    weights_path = Path(root, "weights.csv.bz2")
    save_dict_arrays_to_csv(
        weights_path, weighted_samples["weights"], dtype=dtype, index=index
    )


def save_dict_arrays_to_csv(
    path: Union[str, Path],
    data: Dict[str, Array],
    dtype: type = np.float32,
    index: bool = False,
) -> None:
    data = {k: tensor_to_array(v, dtype=dtype) for k, v in data.items()}
    pd.DataFrame(data).to_csv(path, index=index)


if __name__ == "__main__":
    pass
