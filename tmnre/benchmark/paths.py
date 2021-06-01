from dataclasses import astuple, dataclass
from pathlib import Path

from swyft.types import PathType


@dataclass
class Paths:
    run_yaml: PathType
    log_prob_true_parameters: PathType
    num_constraining_simulations: PathType
    num_simulations_simulator: PathType
    weighted_samples_root: PathType
    samples_root: PathType
    runtime: PathType
    num_elapsed_rounds: PathType
    data_dimension: PathType
    parameter_dimension: PathType
    metrics: PathType
    inference_state_dict: PathType


def prepend_root(root: PathType, paths: Paths) -> Paths:
    root = Path(root)
    args = astuple(paths)
    rooted_args = [root / arg for arg in args]
    return Paths(*rooted_args)


benchmark_paths = Paths(
    run_yaml="run.yaml",
    log_prob_true_parameters="log_prob_true_parameters.csv",
    num_constraining_simulations="num_constraining_simulations.csv",
    num_simulations_simulator="num_simulations_simulator.csv",
    weighted_samples_root="weighted_posterior_samples",
    samples_root="marginal_posterior_samples",
    runtime="runtime.csv",
    num_elapsed_rounds="num_elapsed_rounds.csv",
    data_dimension="data_dim.csv",
    parameter_dimension="parameter_dim.csv",
    metrics="metrics.csv",
    inference_state_dict="inference_state_dict.pt",
)
