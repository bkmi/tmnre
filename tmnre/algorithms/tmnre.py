import logging
from functools import partial
from itertools import combinations
from typing import Optional

import numpy as np
import torch
from toolz import compose

import swyft
import swyft.bounds
from tmnre.nn.resnet import make_resenet_tail
from sbibm.tasks.task import Task
from tmnre.algorithms.priors import get_affine_uniform_prior, get_diagonal_normal_prior, get_diagonal_lognormal_prior


SIMKEY = "X"


def unbatch(tensor):
    assert tensor.shape[0] == 1
    return tensor[0, ...]


def swyftify_observation(x):
    if isinstance(x, torch.Tensor):
        y = x.squeeze().detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        y = x.squeeze()
    else:
        raise NotImplementedError()
    return {SIMKEY: y}


def swyftify_simulator(fn, x):
    xt = torch.from_numpy(x).float()
    y = fn(xt).squeeze().numpy()
    return swyftify_observation(y)


def swyftify_prior(sbibm_task: Task):
    name = sbibm_task.name
    prior_params = sbibm_task.prior_params

    if name == "gaussian_linear":
        prior = get_diagonal_normal_prior(
            prior_params["loc"], prior_params["precision_matrix"],
        )
    elif name == "gaussian_linear_uniform":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "gaussian_correlated_uniform":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "slcp":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "slcp_distractors":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "bernoulli_glm":
        raise NotImplementedError("This prior is not independent.")
    elif name == "bernoulli_glm_raw":
        raise NotImplementedError("This prior is not independent.")
    elif name == "gaussian_mixture":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "two_moons":
        prior = get_affine_uniform_prior(
            prior_params["low"], prior_params["high"], dim=sbibm_task.dim_parameters,
        )
    elif name == "sir":
        prior = get_diagonal_lognormal_prior(
            prior_params["loc"], prior_params["scale"],
        )
    elif name == "lotka_volterra":
        prior = get_diagonal_lognormal_prior(
            prior_params["loc"], prior_params["scale"],
        )
    return prior


def run(
    task: Task,
    num_samples: int,
    num_simulations: int,
    observation: Optional[torch.Tensor] = None,
    num_weighted_samples: Optional[int] = None,
    initial_simulation_factor: float = 0.5,
    compute_2d_marginals: bool = True,
    device: str = "cpu",
    simulation_batch_size: int = 1000,
    max_rounds: int = 10,
    new_simulation_factor: float = 1.0,
    new_simulation_term: int = 0,
    convergence_ratio: float = 0.8,
    neural_net: str = "swyft-default",
    hidden_features: int = 50,
    batch_size: int = 256,
    validation_size: float = 0.10,
    early_stopping_patience: int = 20,
    max_epochs: int = 300,
    lr: float = 1e-2,
    reduce_lr_factor: float = 0.1,
    reduce_lr_patience: int = 5,
    online_z_score_head: bool = True,
    online_z_score_tail: bool = True,
    nworkers: int = 4,
):
    # Args:
    #     task: Task instance
    #     num_samples: Number of samples to generate from posterior
    #     observation: Observation
    #     num_simulations: Simulation budget
    #     max_rounds: Maximum number of rounds
    #     neural_net: Neural network to use, one of linear / mlp / resnet / swyft-default
    #     hidden_features: Number of hidden features in network
    #     simulation_batch_size: Batch size for simulator
    #     batch_size: Batch size for training network
    #     new_simulation_factor: Increase simulations by this factor every round.
    #     online_z_score_head: Use online z_scoring for the head
    #     online_z_score_tail: Use online z_scoring for the tail
    #     convergence_ratio: When the ratio of the new volume to the previous volume is above this number, converged.
    log = logging.getLogger(__name__)
    logging.getLogger("swyft").setLevel(logging.WARNING)

    assert 0.0 < initial_simulation_factor and initial_simulation_factor <= 1.0
    num_initial_simulations = int(num_simulations * initial_simulation_factor)
    num_posterior_samples = num_samples
    if num_weighted_samples is None:
        num_weighted_samples = num_samples

    if simulation_batch_size > num_initial_simulations:
        simulation_batch_size = num_initial_simulations

    if batch_size > num_initial_simulations:
        batch_size = num_initial_simulations

    prior = swyftify_prior(task)
    observation = compose(swyftify_observation, unbatch)(observation)

    sbibm_sim = task.get_simulator(max_calls=None)
    forward = partial(swyftify_simulator, sbibm_sim)  # This is with "baked in noise"
    simulator = swyft.Simulator(forward, sim_shapes={SIMKEY: (task.dim_data,)})
    store = swyft.MemoryStore(task.dim_parameters, simulator=simulator)

    if neural_net in ["linear", "mlp"]:
        raise NotImplementedError
    elif neural_net == "resnet":
        head = swyft.DefaultHead
        head_args = {"online_norm": online_z_score_head}
        tail = make_resenet_tail
        tail_args = dict(
            hidden_features=hidden_features,
            num_blocks=2,
            online_z_score_obs=online_z_score_tail,
            online_z_score_par=online_z_score_tail,
            # dropout_probability=0.0,
            # activation = F.relu,
            use_batch_norm=True,
        )
    elif neural_net == "swyft-default":
        head = swyft.DefaultHead
        head_args = {"online_norm": online_z_score_head}
        tail = swyft.DefaultTail
        tail_args = {
            # "n_tail_features": 2,
            # "p": 0.0,
            "hidden_layers": [hidden_features, hidden_features],
            "online_norm": online_z_score_tail,
            # "param_transform": None,
            # "tail_features": False,
        }

    train_args = dict(
        batch_size=batch_size,
        validation_size=validation_size,
        early_stopping_patience=early_stopping_patience,
        max_epochs=max_epochs,
        lr=lr,
        reduce_lr_factor=reduce_lr_factor,
        reduce_lr_patience=reduce_lr_patience,
        nworkers=nworkers,
        optimizer_fn=torch.optim.Adam,
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    # "zooming in"
    micro = swyft.Microscope(
        [(i,) for i in range(task.dim_parameters)],
        prior,
        observation,
        store=store,
        device=device,
        Ninit=num_initial_simulations,
        train_args=train_args,
        tail=tail,
        tail_args=tail_args,
        head=head,
        head_args=head_args,
        new_simulation_factor=new_simulation_factor,
        new_simulation_term=new_simulation_term,
        convergence_ratio=convergence_ratio,
    )
    log.info("Focusing...")
    micro.focus(max_rounds=max_rounds)
    log.info("Focusing done.")

    # Training reported
    num_constraining_simulations = micro.n_simulations[-1]
    num_remaining_budget = num_simulations - num_constraining_simulations

    num_constraining_simulations_in_region = micro.N[-1]
    num_samples_in_region = (
        num_remaining_budget + num_constraining_simulations_in_region
    )
    constrained_prior = micro.constrained_prior

    dataset = swyft.Dataset(num_samples_in_region, constrained_prior, store,)

    dataset.simulate()
    posterior = swyft.Posteriors(dataset)
    posterior.infer(
        [(i,) for i in range(task.dim_parameters)],
        device=device,
        train_args=train_args,
        tail=tail,
        tail_args=tail_args,
        head=head,
        head_args=head_args,
    )

    if compute_2d_marginals:
        log.info("Two dimensional marginals.")
        posterior.infer(
            list(combinations(range(task.dim_parameters), 2)),
            device=device,
            train_args=train_args,
            tail=tail,
            tail_args=tail_args,
            head=head,
            head_args=head_args,
        )

    if False:
        weighted_samples = [
            p.sample(num_weighted_samples, observation) for p in micro.posteriors
        ]
        samples = [
            p.rejection_sample(num_posterior_samples, observation)
            for p in micro.posteriors
        ]

        state_dicts = [p.state_dict() for p in micro.posteriors]
    else:
        log.info("Getting weighted samples")
        weighted_samples = posterior.sample(
            num_weighted_samples, observation, device=device, n_batch=10_000
        )
        log.info("Getting posterior samples")
        samples = posterior.rejection_sample(
            num_posterior_samples, observation, device=device, n_batch=10_000
        )
        state_dicts = posterior.state_dict()

    return (
        weighted_samples,
        samples,
        len(store),
        len(dataset),  # num training simulations for reported results
        micro.elapsed_rounds,
        state_dicts,
    )


def main(task_name):
    import sys

    import sbibm
    from sbibm.utils.debug import pdb_hook

    sys.excepthook = pdb_hook

    task = sbibm.get_task(task_name, dim=5,)
    observation = task.get_observation(1)
    out = run(
        task=task,
        num_samples=10_000,
        num_simulations=1_000,
        observation=observation,
        num_weighted_samples=1_000_000,
        # max_rounds=1,
        # max_epochs=10,
        # device="cuda:0",
    )
    print(out)


if __name__ == "__main__":
    tasks = [
        "bernoulli_glm",
        "gaussian_mixture",
        "two_moons",
        "bernoulli_glm_raw",
        "slcp",
        "lotka_volterra",
        "sir",
        "slcp_distractors",
        "gaussian_linear_uniform",
        "gaussian_linear",
    ]

    main(task_name="gaussian_linear_uniform")
