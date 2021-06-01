# pylint: disable=no-value-for-parameter
import logging
import os
import socket
import sys
from time import perf_counter

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from toolz import compose
from toolz.curried import keyfilter, keymap

import sbibm
from tmnre.benchmark import (
    get_estimated_posterior_samples,
    get_folder,
    get_reference_posterior_samples,
)
from tmnre.io import save_config, save_yaml
from tmnre.marginalize import check_if_x_dim
from tmnre.metrics.c2st import c2sts
from sbibm.utils.debug import pdb_hook


def c2st_mean_var(c2sts: dict) -> dict:
    mean_c2st = np.mean(list(c2sts.values()))
    var_c2st = np.var(list(c2sts.values()))
    return {"mean": float(mean_c2st), "var": float(var_c2st)}


def c2st_stats_by_dim(c2sts_marginals: dict) -> dict:
    out = {}
    for dimension, predicate in check_if_x_dim.items():
        stats = compose(c2st_mean_var, keyfilter(predicate))(c2sts_marginals)
        out.update(keymap(lambda key: f"{dimension} {key}", stats))
    return out


@hydra.main(config_path="c2st_config", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.debug:
        sys.excepthook = pdb_hook

    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"sbibm version: {sbibm.__version__}")
    log.info(f"Hostname: {socket.gethostname()}")
    if cfg.seed is None:
        log.info("Seed not specified, generating random seed for replicability")
        cfg.seed = int(np.random.randint(low=1, high=2 ** 32 - 1, size=(1,))[0])
        log.info(f"Random seed: {cfg.seed}")
    start = perf_counter()
    save_config(cfg)

    raw_results_root = hydra.utils.to_absolute_path(cfg.raw_results_root)

    results_df = sbibm.get_results(dataset="main_paper.csv")
    folder = get_folder(
        results_df,
        cfg.task.name,
        cfg.task.num_observation,
        cfg.task.num_simulations,
        cfg.algorithm.name,
    )

    log.info(
        f"Loading estimated samples from {os.path.join(raw_results_root, folder)}"
    )
    estimated_samples = get_estimated_posterior_samples(
        folder, raw_results_root
    )
    log.info(
        f"Loading reference samples from {os.path.join(raw_results_root, folder)}"
    )
    reference_samples = get_reference_posterior_samples(
        cfg.task.name, cfg.task.num_observation
    )

    log.info("Training c2st on each marginal")
    all_c2sts = c2sts(
        estimated_samples, reference_samples, n_jobs=cfg.task.tasks_per_node
    )
    save_yaml("marginals.yaml", all_c2sts)

    log.info("Calculating c2st stats")
    c2sts_stats = c2st_stats_by_dim(all_c2sts)
    save_yaml("stats.yaml", {k: v for k, v in c2sts_stats.items()})

    log.info(f"Duration (mins): {(perf_counter() - start) / 60}")


if __name__ == "__main__":
    main()
