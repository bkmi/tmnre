import logging
import pickle
from functools import partial
from itertools import chain, product

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

TASK_NAME = "torus"
NUM_OBS = 1
DIM = 3
N_SIMULATIONS = [4_985, 11_322, 21_127, 32_032]
N_POSTERIOR_SAMPLES = 50_000
NUM_WORKERS = 4
NUM_CORES = 16
N_JOBS = NUM_CORES // NUM_WORKERS

task = sbibm.get_task(
    TASK_NAME,
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


for ns in N_SIMULATIONS:
    non = do_it(
        partial(get_posterior_samples, n_sims=ns),
        ["NPE", "NLE", "NRE"],
        "torus-sbi-non-{ns:06d}.pickle",
    )


############################
#### Sequential
############################


def get_sequential_posterior_samples(
    sim,
    pri,
    inference,
    num_workers=NUM_WORKERS,
    n_sims: list = N_SIMULATIONS,
):
    if inference == "SNRE" or inference == "NRE":
        inference = sbi.inference.SNRE_A(prior=sbi_prior)
    elif inference == "SNPE" or inference == "NPE":
        inference = sbi.inference.SNPE_C(prior=sbi_prior)
    elif inference == "SNLE" or inference == "NLE":
        inference = sbi.inference.SNLE_A(prior=sbi_prior)
    else:
        raise NotImplementedError()

    posteriors = []
    proposal = pri
    for r, ns in enumerate(n_sims):
        theta, x = sbi.inference.simulate_for_sbi(
            sim, proposal, num_simulations=ns, num_workers=num_workers
        )
        density_estimator = inference.append_simulations(theta, x, r).train()
        posterior = inference.build_posterior(
            density_estimator,
        )
        posteriors.append(posterior)
        proposal = posterior.set_default_x(torch.atleast_2d(obs0))
    return posteriors[1:]


def sample_and_save(posterior, n_post_sims, suffix):
    path = f"torus-sbi-{suffix}.pickle"
    if SAVE:
        samples = posterior.sample((int(n_post_sims),), x=torch.atleast_2d(obs0))
        with open(path, "wb") as f:
            pickle.dump(samples, f)
    else:
        with open(path, "rb") as f:
            samples = pickle.load(f)
    return samples


def seq_do_it(func, methods, n_post_sims=N_POSTERIOR_SAMPLES):
    suffixes = list(
        map(
            lambda x: f"{x[0]}-{x[1]:06d}",
            product(methods, N_SIMULATIONS[1:]),
        )
    )

    with parallel_backend("loky", inner_max_num_threads=NUM_WORKERS):
        posteriors = Parallel(n_jobs=N_JOBS)(
            delayed(func)(sbi_simulator, sbi_prior, inference_method)
            for inference_method in methods
        )

    posteriors = list(chain(*posteriors))

    with parallel_backend("loky", inner_max_num_threads=1):
        posterior_samples = Parallel(n_jobs=N_JOBS)(
            delayed(sample_and_save)(posterior, n_post_sims, suffix)
            for posterior, suffix in zip(posteriors, suffixes)
        )

    return posterior_samples


seq_do_it(get_sequential_posterior_samples, ["NPE", "NLE", "NRE"])
