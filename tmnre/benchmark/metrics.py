import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

import sbibm
from tmnre.benchmark.paths import Paths, benchmark_paths
from tmnre.functional.dicttools import filter_by_key_intersection
from tmnre.io import load_marginal_samples_from_csv_directory
from tmnre.marginalize import all_limits, all_marginals
from tmnre.metrics import c2st_on_marginals, kl_on_marginals  # noqa: F401
from sbibm.utils.io import get_float_from_csv, get_int_from_csv  # noqa: F401


def denest_marginal_metrics(metrics_dict):
    denested_metrics = {}
    for metric, result in metrics_dict.items():
        if isinstance(result, dict):
            for key, value in result.items():
                denested_metrics[f"{metric} {key}"] = value
        else:
            denested_metrics[metric] = result
    return denested_metrics


def compute_metrics_df(
    task: Union[sbibm.tasks.task.Task],
    num_observation: int,
    benchmark_paths: Paths = benchmark_paths,
    log: logging.Logger = logging.getLogger(__name__),
    n_jobs: Optional[int] = None,
    do_kld: bool = True,
) -> pd.DataFrame:
    """Compute all metrics, returns dataframe

    Args:
        task_name: Task
        num_observation: Observation
        path_samples: Path to posterior samples
        path_runtime: Path to runtime file
        log: Logger

    Returns:
        Dataframe with results
    """
    log.info("Compute all metrics")

    # Load task
    if isinstance(task, sbibm.tasks.task.Task):
        pass
    elif isinstance(task, str):
        raise NotImplementedError(
            "We need to report the dim, scale, etc of the posterior to get the correct task."
        )
    else:
        raise NotImplementedError

    # Load limits for kl div
    if do_kld:
        task_limits = task.get_param_limits().numpy()
        # task_names = task.get_names_parameters()
        # assert len(task_limits) == len(task_names)
        # limits = all_limits(task_limits, task_names)  # noqa: F841
        limits = all_limits(task_limits)  # noqa: F841

    # Load samples
    reference_posterior_samples = task.get_reference_posterior_samples(num_observation)[
        : task.num_posterior_samples, :
    ]

    reference_marginalized_posterior_samples = all_marginals(
        reference_posterior_samples
    )

    algorithm_marginalized_posterior_samples = load_marginal_samples_from_csv_directory(
        benchmark_paths.samples_root
    )

    log.debug(f"alg, {algorithm_marginalized_posterior_samples.keys()}")
    log.debug(f"ref, {reference_marginalized_posterior_samples.keys()}")
    log.info(
        f"Keys difference between yours and all 1d, 2d marginals:  {set(reference_marginalized_posterior_samples.keys()) - set(algorithm_marginalized_posterior_samples.keys())}"
    )

    (
        algorithm_marginalized_posterior_samples,
        reference_marginalized_posterior_samples,
    ) = filter_by_key_intersection(
        algorithm_marginalized_posterior_samples,
        reference_marginalized_posterior_samples,
    )

    assert all(
        x.shape[0] == task.num_posterior_samples
        for x in reference_marginalized_posterior_samples.values()
    )
    assert all(
        x.shape[0] == task.num_posterior_samples
        for x in algorithm_marginalized_posterior_samples.values()
    )
    log.info(
        f"Loaded {task.num_posterior_samples} samples from reference and algorithm"
    )

    # how long did it take?
    runtime_sec = np.atleast_1d(  # noqa: F841
        get_float_from_csv(benchmark_paths.runtime)
    )

    # Names of all metrics as keys, values are calls that are passed to eval
    _METRICS_ = {
        "C2ST": "c2st_on_marginals(X=reference_marginalized_posterior_samples, Y=algorithm_marginalized_posterior_samples, z_score=True, n_jobs=n_jobs)",
        #
        # Not based on samples
        #
        # "NLTP": "-1. * log_prob_true_parameters",
        "RT": "runtime_sec",
        # "N_RNDs": "np.atleast_1d(get_int_from_csv(metric_paths['num_elapsed_rounds']))",
        # "data_dim": "np.atleast_1d(get_int_from_csv(metric_paths['data_dimension']))",
        # "parameter_dim": "np.atleast_1d(get_int_from_csv(metric_paths['parameter_dimension']))",
    }

    if do_kld:
        _METRICS_["KLD"] = "kl_on_marginals(X=reference_marginalized_posterior_samples, Y=algorithm_marginalized_posterior_samples, limits=limits, bins=10, add_a_sample=False, n_jobs=n_jobs)",
        _METRICS_["KLD_FIX"] = "kl_on_marginals(X=reference_marginalized_posterior_samples, Y=algorithm_marginalized_posterior_samples, limits=limits, bins=10, add_a_sample=True, n_jobs=n_jobs)",

    metrics_dict = {}
    for metric, eval_cmd in _METRICS_.items():
        log.info(f"Computing {metric}")
        try:
            metrics_dict[metric] = eval(eval_cmd)
            log.info(f"{metric}: {metrics_dict[metric]}")
        except AssertionError:
            log.info(f"{metric} had an assertion error.")
        except:  # noqa: E722
            metrics_dict[metric] = float("nan")

    marginal_metrics = denest_marginal_metrics(metrics_dict)

    return pd.DataFrame(marginal_metrics)
