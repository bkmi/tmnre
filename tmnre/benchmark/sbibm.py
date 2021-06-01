import os
from typing import Dict

import numpy as np
import pandas as pd
from toolz import valmap

from tmnre.marginalize import all_marginals
from sbibm import get_results, get_task


num_simulations_int_to_str = {
    1000: "10³",
    10000: "10⁴",
    100000: "10⁵",
}


def translate_num_simulations(num_simulations: int):
    return num_simulations_int_to_str[num_simulations]


def get_estimated_posterior_samples(folder, root):
    return pd.read_csv(
        os.path.join(root, folder, "posterior_samples.csv.bz2")
    ).to_numpy()


def get_reference_posterior_samples(task_name, num_observation):
    return (
        get_task(task_name)
        .get_reference_posterior_samples(num_observation=num_observation)
        .numpy()
    )


def get_folder(
    df: pd.DataFrame,
    task_name: str,
    num_observation: int,
    num_simulations: int,
    algorithm: str,
):
    if isinstance(num_simulations, int):
        num_simulations = translate_num_simulations(num_simulations)

    task_name = task_name.lower()
    algorithm = algorithm.upper()

    assert task_name in df["task"].unique()
    assert num_observation in df["num_observation"].unique()
    assert num_simulations in df["num_simulations"].unique()
    assert algorithm in df["algorithm"].unique()

    row = df[
        (df["task"] == task_name)
        & (df["num_observation"] == num_observation)
        & (df["num_simulations"] == num_simulations)
        & (df["algorithm"] == algorithm)
    ]
    return row["folder"].item()


def load_ref_and_marginalize(task, num_observation: int) -> Dict[str, np.ndarray]:
    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)[
        : task.num_posterior_samples, :
    ]

    reference_marginalized_posterior_samples = all_marginals(
        reference_posterior_samples,
        # task.get_names_parameters(),
    )
    reference_marginalized_posterior_samples = valmap(
        lambda x: x.numpy(), reference_marginalized_posterior_samples
    )
    return reference_marginalized_posterior_samples


def load_alg_and_marginalize(
    task_name: str,
    num_observation: int,
    algorithm: str,
    num_simulations: int,
    root: str,
) -> Dict[str, np.ndarray]:
    results_df = get_results(dataset="main_paper.csv")
    folder = get_folder(
        results_df, task_name, num_observation, num_simulations, algorithm
    )
    estimated_samples = get_estimated_posterior_samples(folder, root)
    estimated_marginalized_posterior_samples = all_marginals(
        estimated_samples, get_task(task_name).get_names_parameters()
    )
    return estimated_marginalized_posterior_samples


if __name__ == "__main__":
    pass
