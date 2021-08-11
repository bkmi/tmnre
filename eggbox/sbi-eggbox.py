import logging
import pickle

import numpy as np
import sbi
import sbi.analysis
import sbi.inference
import sbi.utils
import torch
from joblib import Parallel, delayed, parallel_backend

import sbibm
import tmnre
import tmnre.benchmark
import tmnre.coverage.oned
import tmnre.metrics

log = logging.getLogger()
log.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

np.random.seed(28)
torch.manual_seed(28)

SAVE = True

TASK_NAME = "eggbox"
NUM_OBS = 1
DIM = 10
N_SIMULATIONS = 10_000
N_POSTERIOR_SAMPLES = 25_000
NUM_WORKERS = 4
NUM_CORES = 16
N_JOBS = NUM_CORES // NUM_WORKERS

task = sbibm.get_task(
    TASK_NAME,
    dim=DIM,
)

theta0 = task.get_true_parameters(NUM_OBS).squeeze()
obs0 = task.get_observation(NUM_OBS).squeeze()
print(theta0)
print(obs0)

simulator = task.get_simulator()
prior_min = torch.zeros(DIM)
prior_max = torch.ones(DIM)
prior = sbi.utils.torchutils.BoxUniform(
    low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max)
)
sbi_simulator, sbi_prior = sbi.inference.prepare_for_sbi(simulator, prior)


def get_sequential_posterior_samples(
    sim,
    pri,
    inference,
    num_workers=NUM_WORKERS,
    n_sims=N_SIMULATIONS,
    n_post_sims=N_POSTERIOR_SAMPLES,
):
    if inference == "SNRE" or inference == "NRE":
        inference = sbi.inference.SNRE_A(prior=sbi_prior)
    elif inference == "SNPE" or inference == "NPE":
        inference = sbi.inference.SNPE_C(prior=sbi_prior)
    elif inference == "SNLE" or inference == "NLE":
        inference = sbi.inference.SNLE_A(prior=sbi_prior)
    else:
        raise NotImplementedError()

    num_rounds = 5
    n_per_round = n_sims // num_rounds
    posteriors = []
    proposal = pri
    for r in range(num_rounds):
        theta, x = sbi.inference.simulate_for_sbi(
            sim, proposal, num_simulations=n_per_round, num_workers=num_workers
        )
        density_estimator = inference.append_simulations(theta, x, r).train()
        posterior = inference.build_posterior(
            density_estimator,
        )
        posteriors.append(posterior)
        proposal = posterior.set_default_x(torch.atleast_2d(obs0))
    return posterior.sample((int(n_post_sims),), x=torch.atleast_2d(obs0))


def get_posterior_samples(
    sim,
    pri,
    inference,
    num_workers=NUM_WORKERS,
    n_sims=N_SIMULATIONS,
    n_post_sims=N_POSTERIOR_SAMPLES,
):
    if inference == "SNRE" or inference == "NRE":
        inference = sbi.inference.SNRE_A(prior=sbi_prior)
        kwargs = {}
    elif inference == "SNPE" or inference == "NPE":
        inference = sbi.inference.SNPE_C(prior=sbi_prior)
        kwargs = {"sample_with_mcmc": True}
    elif inference == "SNLE" or inference == "NLE":
        inference = sbi.inference.SNLE_A(prior=sbi_prior)
        kwargs = {}
    else:
        raise NotImplementedError()

    theta, x = sbi.inference.simulate_for_sbi(
        sim, pri, num_simulations=n_sims, num_workers=num_workers
    )
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator, **kwargs)
    return posterior.sample((int(n_post_sims),), x=torch.atleast_2d(obs0))


def do_it(func, methods, path):
    with parallel_backend("loky", inner_max_num_threads=NUM_WORKERS):
        result = Parallel(n_jobs=N_JOBS)(
            delayed(func)(sbi_simulator, sbi_prior, inference_method)
            for inference_method in methods
        )

    posterior_samples = {method: samples for method, samples in zip(methods, result)}

    if SAVE:
        with open(path, "wb") as f:
            pickle.dump(posterior_samples, f)
    else:
        with open(path, "rb") as f:
            posterior_samples = pickle.load(f)

    return posterior_samples


seq = do_it(
    get_sequential_posterior_samples,
    [
        "SNPE",
    ],
    "eggbox-sbi-snpe.pickle",
)

non = do_it(get_posterior_samples, ["NPE", "NLE"], "eggbox-sbi-non.pickle")

seq = do_it(
    get_sequential_posterior_samples,
    [
        "SNLE",
    ],
    "eggbox-sbi-snle.pickle",
)
