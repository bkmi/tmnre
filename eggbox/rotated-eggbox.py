import logging
import pickle
from functools import partial
from itertools import combinations, product
from typing import Hashable

import numpy as np
import sbi
import sbi.analysis
import sbi.inference
import sbi.utils
import torch
from joblib.parallel import Parallel, delayed, parallel_backend
from toolz.functoolz import compose

import sbibm
import swyft
import tmnre
import tmnre.benchmark
import tmnre.coverage.oned
import tmnre.metrics
from tmnre.nn.resnet import make_resenet_tail

log = logging.getLogger()
log.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_rotation_matrix(d: int) -> np.ndarray:
    # We aim to rotate x to y
    x = np.ones((d, 1))
    y = np.zeros((d, 1))
    y[-1, :] = 1.0

    # Create orthonormal basis for the planar subspace
    u = x / np.linalg.norm(x)
    v = y - np.dot(u.T, y) * u
    v /= np.linalg.norm(v)

    # Create rotation matrix within planar subspace
    cos_theta = np.dot(x.T, y).flatten() / (np.linalg.norm(x) * np.linalg.norm(y))
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    R_theta = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]).squeeze()
    uv = np.concatenate([u, v], axis=-1)

    # Assemble total rotation (I - uu' - vv') + ([u, v] R [u, v].T) == Project onto complement + project onto rotated planar subspace
    return np.eye(d) - np.dot(u, u.T) - np.dot(v, v.T) + uv @ R_theta @ uv.T


def get_new_uniform_bounds(q, d):
    box_corners = np.asarray(list(product([0.0, 1.0], repeat=d))).T
    upper_bounds = (q @ box_corners).max(axis=1).tolist()
    lower_bounds = (q @ box_corners).min(axis=1).tolist()
    bounds = list(zip(lower_bounds, upper_bounds))
    return bounds


def swyftify_simulator(fn, x: np.ndarray, simkey: Hashable):
    xt = torch.from_numpy(x).float()
    y = fn(xt).squeeze().numpy()
    return {simkey: y}


def get_prior_transform(bounds: np.ndarray):
    """bounds.shape == (dim, 2)"""
    bounds = np.asarray(bounds)
    length = np.diff(bounds, axis=-1).squeeze()
    left = np.min(bounds, axis=-1)

    def prior_transform(theta):
        return theta * length + left

    return prior_transform


def inverse_rotate_params(theta: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Goes inside swyftify simulator, therefore tensorfloat to tensorfloat on the cpu."""
    return q.t() @ theta


def get_hps(myhps: bool = True):
    if myhps:
        train_args = dict(
            batch_size=256,
            validation_size=0.10,
            early_stopping_patience=20,
            max_epochs=300,
            lr=1e-2,
            reduce_lr_factor=0.1,
            reduce_lr_patience=5,
            nworkers=3,
            # optimizer_fn=partial(torch.optim.SGD, momentum=0.9)
        )
        tail_args = dict(
            # hidden_features=128 + 32,
            hidden_features=50,
            num_blocks=2,
            online_z_score_obs=True,
            online_z_score_par=True,
            # dropout_probability=0.0,
            # activation = F.relu,
            use_batch_norm=True,
        )
        head_args = {
            # 'obs_shapes': {'mu': torch.Size([3])},
            # 'obs_transform': None,
            "online_norm": False,
        }
        get_tail = make_resenet_tail
    else:
        train_args = {}
        tail_args = {}
        head_args = {}
        get_tail = swyft.DefaultTail
    return train_args, tail_args, head_args, get_tail


def sample_sbi(posterior, n_posterior_samples: int, obs: torch.Tensor) -> torch.Tensor:
    return posterior.sample((int(n_posterior_samples),), x=torch.atleast_2d(obs))


def main():
    # Setup
    np.random.seed(28)
    torch.manual_seed(28)

    TASK_NAME = "eggbox"
    NUM_OBS = 1
    DIM = 10
    N_SIMULATIONS = [1_000, 10_000, 100_000]
    N_POSTERIOR_SAMPLES = 25_000
    DEVICE = torch.device("cuda:0")
    SIMKEY = "mu"

    task = sbibm.get_task(
        TASK_NAME,
        dim=DIM,
    )

    theta0 = task.get_true_parameters(NUM_OBS).squeeze()
    obs0 = task.get_observation(NUM_OBS).squeeze()
    print(f"{theta0=}")
    print(f"{obs0=}")

    Q = get_rotation_matrix(DIM)
    Qt = torch.from_numpy(Q).float()
    new_bounds = get_new_uniform_bounds(Q, DIM)
    prior_transform = get_prior_transform(new_bounds)

    invr = partial(inverse_rotate_params, Qt)
    rotated_simulator = compose(task.get_simulator(), invr)

    print(f"{Qt @ theta0=}")

    forward = partial(
        swyftify_simulator, rotated_simulator, simkey=SIMKEY
    )
    simulator = swyft.Simulator(forward, sim_shapes={SIMKEY: (task.dim_data,)})
    store = swyft.MemoryStore(task.dim_parameters, simulator=simulator)

    prior = swyft.Prior.from_uv(prior_transform, task.dim_parameters)
    theta0_swyft = prior_transform(theta0.clone().squeeze().float().numpy())
    obs0_swyft = {SIMKEY: obs0.clone().squeeze().float().numpy()}
    print(obs0_swyft)

    twod_marginals = list(combinations(range(task.dim_parameters), 2))
    print(f"list of 2d marginals: {twod_marginals}")

    train_args, tail_args, head_args, get_tail = get_hps()
    sbi_prior = sbi.utils.torchutils.BoxUniform(
        low=torch.tensor(new_bounds)[:, 0],
        high=torch.tensor(new_bounds)[:, 1],
    )

    # #################
    # # swyft
    # #################
    swyft_datasets = []
    swyft_posts = []
    swyft_samples = []
    swyft_rejection_samples = []
    for n in N_SIMULATIONS:
        dataset = swyft.Dataset(
            n,
            prior,
            store,
        )
        dataset.simulate()
        posterior = swyft.Posteriors(dataset)
        posterior.infer(
            [(i,) for i in range(task.dim_parameters)],
            device=DEVICE,
            train_args=train_args,
            tail=get_tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        posterior.infer(
            twod_marginals,
            device=DEVICE,
            train_args=train_args,
            tail=get_tail,
            tail_args=tail_args,
            head_args=head_args,
        )
        samples = posterior.sample(N_POSTERIOR_SAMPLES, obs0_swyft)
        rejection_samples = posterior.rejection_sample(
            N_POSTERIOR_SAMPLES,
            obs0_swyft,
            excess_factor=10,
        )

        swyft_datasets.append(dataset)
        swyft_posts.append(posterior)
        swyft_samples.append(samples)
        swyft_rejection_samples.append(rejection_samples)

    swyft_samples_path = "rot-swyft-samples.pickle"
    swyft_prefix = "swyft/nsims-"
    swyft_rejection_prefix = "rej-swyft/nsims-"

    swyft_payload = {
        swyft_prefix + str(n): s for n, s in zip(N_SIMULATIONS, swyft_samples)
    }
    swyft_payload.update(
        {
            swyft_rejection_prefix + str(n): s
            for n, s in zip(N_SIMULATIONS, swyft_rejection_samples)
        }
    )
    with open(swyft_samples_path, "wb") as f:
        pickle.dump(swyft_payload, f)

    #################
    # sbi
    #################
    sbi_posteriors = []
    for n in N_SIMULATIONS:
        posterior = sbi.inference.base.infer(
            rotated_simulator,
            sbi_prior,
            method="SNRE_A",  # SNLE_A
            num_simulations=n,
            num_workers=4,
        )
        sbi_posteriors.append(posterior)

    with parallel_backend("loky", inner_max_num_threads=2):
        sbi_samples = Parallel(n_jobs=len(N_SIMULATIONS))(
            delayed(sample_sbi)(post, n, obs0)
            for post, n in zip(sbi_posteriors, N_SIMULATIONS)
        )
    sbi_samples = [s.numpy() for s in sbi_samples]

    sbi_samples_path = "rot-sbi-samples.pickle"
    sbi_prefix = "sbi/nsims-"
    sbi_payload = {sbi_prefix + str(n): s for n, s in zip(N_SIMULATIONS, sbi_samples)}
    with open(sbi_samples_path, "wb") as f:
        pickle.dump(sbi_payload, f)

    #################
    # sequentially
    #################
    seq_posteriors = []
    seq_samples = []

    seq_sim, seq_prior = sbi.inference.prepare_for_sbi(
        rotated_simulator,
        sbi_prior,
    )

    for n in N_SIMULATIONS:
        num_rounds = 5
        n_per_round = n // num_rounds

        posteriors = []
        inference = sbi.inference.SNRE_A(prior=seq_prior)
        proposal = seq_prior
        for r in range(num_rounds):
            theta, x = sbi.inference.simulate_for_sbi(seq_sim, proposal, num_simulations=n_per_round, num_workers=4)
            density_estimator = inference.append_simulations(theta, x, from_round=r).train()
            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            proposal = posterior.set_default_x(torch.atleast_2d(obs0))
        samples = posterior.sample(
            (int(N_POSTERIOR_SAMPLES),),
            x=torch.atleast_2d(obs0)
        )
        seq_posteriors.append(posterior)
        seq_samples.append(samples)

    seq_samples = [s.numpy() for s in seq_samples]
    seq_samples_path = "rot-seq-samples.pickle"
    seq_prefix = "seq/nsims-"
    seq_payload = {seq_prefix + str(n): s for n, s in zip(N_SIMULATIONS, seq_samples)}
    with open(seq_samples_path, "wb") as f:
        pickle.dump(seq_payload, f)
    
    #################
    # SNLE
    #################
    snle_posteriors = []
    snle_samples = []

    snle_sim, snle_prior = sbi.inference.prepare_for_sbi(
        rotated_simulator,
        sbi_prior,
    )

    for n in N_SIMULATIONS:
        num_rounds = 5
        n_per_round = n // num_rounds

        posteriors = []
        inference = sbi.inference.SNLE_A(prior=snle_prior)
        proposal = snle_prior
        for r in range(num_rounds):
            theta, x = sbi.inference.simulate_for_sbi(snle_sim, proposal, num_simulations=n_per_round, num_workers=4)
            density_estimator = inference.append_simulations(theta, x, from_round=r).train()
            posterior = inference.build_posterior(density_estimator)
            posteriors.append(posterior)
            proposal = posterior.set_default_x(torch.atleast_2d(obs0))
        samples = posterior.sample(
            (int(N_POSTERIOR_SAMPLES),),
            x=torch.atleast_2d(obs0)
        )
        snle_posteriors.append(posterior)
        snle_samples.append(samples)

    snle_samples = [s.numpy() for s in snle_samples]
    snle_samples_path = "rot-snle-samples.pickle"
    snle_prefix = "snle/nsims-"
    snle_payload = {snle_prefix + str(n): s for n, s in zip(N_SIMULATIONS, snle_samples)}
    with open(snle_samples_path, "wb") as f:
        pickle.dump(snle_payload, f)

if __name__ == "__main__":
    main()
