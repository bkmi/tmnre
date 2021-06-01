import importlib
import logging
import random
import socket
import sys
import time
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf.errors import ConfigAttributeError
from pkg_resources import get_distribution
from toolz import compose, keyfilter
from toolz.curried import keymap

import sbibm
from tmnre.benchmark.metrics import compute_metrics_df
from tmnre.benchmark.paths import benchmark_paths
from tmnre.io import (
    save_config,
    save_marginal_samples_to_csv_directory,
    save_weighted_samples_to_csv_directory,
)
from tmnre.marginalize import all_marginals
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import save_float_to_csv, save_int_to_csv


def create_observation(simulator, prior_dist):
    parameters = prior_dist.sample()
    return simulator(parameters)


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.debug:
        sys.excepthook = pdb_hook

    log = logging.getLogger(__name__)
    log.info(cfg.pretty())
    log.info(f"swyft version: {get_distribution('swyft').version}")
    log.info(f"sbibm version: {sbibm.__version__}")
    log.info(f"Hostname: {socket.gethostname()}")
    if cfg.seed is None:
        log.info("Seed not specified, generating random seed for replicability")
        cfg.seed = int(torch.randint(low=1, high=2 ** 32 - 1, size=(1,))[0])
        log.info(f"Random seed: {cfg.seed}")
    save_config(cfg)

    # Seeding
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Devices
    gpu = True if cfg.hardware.device != "cpu" else False
    if gpu and cfg.algorithm.name != "tmnre":
        torch.cuda.set_device(0)  # type: ignore
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if gpu else "torch.FloatTensor"
        )

    # Run
    log.info(f"get_task params {cfg.task.params}")
    task = sbibm.get_task(cfg.task.name, **cfg.task.params)

    # No observation selected? Make one. Otherwise, use that observation.
    if cfg.task.num_observation is None:
        observation = create_observation(task.get_simulator(), task.get_prior_dist())
    else:
        observation = task.get_observation(cfg.task.num_observation)

    t0 = time.time()
    parts = cfg.algorithm.run.split(".")
    module_name = ".".join(parts[:-1])
    run_fn = getattr(importlib.import_module(module_name), parts[-1])
    algorithm_params = cfg.algorithm.params if "params" in cfg.algorithm else {}
    log.info("Start run")
    outputs = run_fn(
        task,
        num_samples=task.num_posterior_samples,
        num_simulations=cfg.task.num_simulations,
        observation=observation,
        **algorithm_params,
    )
    runtime = time.time() - t0
    log.info("Finished run")

    # Store outputs
    if isinstance(outputs, tuple) and len(outputs) == 3:
        raise NotImplementedError("The script does not work on non-swyft sbi algorithms. Consult the sbibm github repo to do those runs.")
        # Not for swyft
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )

        def string_to_paramemeter_tuple(string: str) -> Tuple[str]:
            return tuple(string.split(","))

        # TODO this needs to be updated to the new parameter methods.
        marginal_samples = compose(keymap(string_to_paramemeter_tuple), all_marginals)(
            samples.cpu().numpy(), param_names=task.get_names_parameters()
        )

        # deal with extra crap needed by swyft runs
        inference_state_dict = {}
        if not cfg.analysis.compute_2d_marginals:

            def is_1d_parameter(parameter: Tuple[str]) -> bool:
                assert isinstance(parameter, tuple)
                if len(parameter) == 1:
                    return True
                else:
                    return False

            marginal_samples = keyfilter(is_1d_parameter, marginal_samples)
        weighted_samples = {"all": np.array([np.nan])}
        try:
            num_elapsed_rounds = cfg.task.params.max_rounds
        except ConfigAttributeError:
            num_elapsed_rounds = 0
        num_constraining_simulations = 0
    elif isinstance(outputs, tuple) and len(outputs) == 6:
        # For swyft
        (
            weighted_samples,
            marginal_samples,
            num_constraining_simulations,
            num_simulations_simulator,
            num_elapsed_rounds,
            inference_state_dict,
        ) = outputs

        # deal with extra crap needed by non-swyft runs
        log_prob_true_parameters = float("nan")
    else:
        raise NotImplementedError

    # maybe this could be done by having a dictionary of functions which are then applied to each thing based off type?
    save_weighted_samples_to_csv_directory(
        benchmark_paths.weighted_samples_root, weighted_samples
    )
    save_marginal_samples_to_csv_directory(
        benchmark_paths.samples_root, marginal_samples
    )
    save_float_to_csv(benchmark_paths.runtime, runtime)
    save_int_to_csv(benchmark_paths.num_elapsed_rounds, num_elapsed_rounds)
    save_int_to_csv(benchmark_paths.data_dimension, task.dim_data)
    save_int_to_csv(benchmark_paths.parameter_dimension, task.dim_parameters)
    save_float_to_csv(
        benchmark_paths.num_constraining_simulations, num_constraining_simulations
    )
    save_float_to_csv(
        benchmark_paths.num_simulations_simulator, num_simulations_simulator
    )
    save_float_to_csv(
        benchmark_paths.log_prob_true_parameters, log_prob_true_parameters
    )
    torch.save(inference_state_dict, benchmark_paths.inference_state_dict)

    # Compute metrics
    if cfg.task.num_observation is None:
        log.info(
            "Cannot compute metrics as reference is unknown. (No num_observation set)"
        )
    elif cfg.analysis.compute_metrics:
        df_metrics = compute_metrics_df(
            task=task,
            num_observation=cfg.task.num_observation,
            benchmark_paths=benchmark_paths,
            log=log,
            n_jobs=cfg.analysis.metric_n_jobs,
        )
        df_metrics.to_csv(benchmark_paths.metrics, index=False)
        log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")


if __name__ == "__main__":
    main()
