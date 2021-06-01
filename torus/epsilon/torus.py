import logging
from functools import partial
from itertools import combinations

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from toolz import identity, valmap

import tmnre
import tmnre.benchmark
import tmnre.io
import tmnre.metrics
import sbibm
import swyft
from tmnre.nn.resnet import make_resenet_tail
from sbibm.utils.io import save_int_to_csv


def get_10k(x):
    return np.random.permutation(x)[:10000, ...]


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(filename="torus.log", filemode="w", level=logging.DEBUG)
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("swyft").setLevel(logging.WARNING)

    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_int_to_csv("seed.csv", seed)

    TASK_NAME = "torus"
    task = sbibm.get_task(
        TASK_NAME,
    )

    NUM_OBS = 1
    NINIT = 10_000
    N_POSTERIOR_SAMPLES = 50_000
    MAX_ROUNDS = 3
    CONVERGENCE_RATIO = 1.0
    EPSILON = np.log(10 ** float(cfg.epsilon))
    SIMKEY = "x"
    DEVICE = torch.device(cfg.hardware.device)

    log.info(f"epsilon threshold: {np.exp(EPSILON):.2E}")

    # theta0 = task.get_true_parameters(NUM_OBS).squeeze()
    obs0 = task.get_observation(NUM_OBS).squeeze()

    def swyftify_simulator(fn, x):
        xt = torch.from_numpy(x).float()
        y = fn(xt).squeeze().numpy()
        return {SIMKEY: y}

    prior_transform = identity

    def swyftify_theta(theta):
        return prior_transform(theta.clone().squeeze().float().numpy())

    def swyftify_obs(obs):
        return {SIMKEY: obs.clone().squeeze().float().numpy()}

    forward = partial(swyftify_simulator, task.get_simulator())
    simulator = swyft.Simulator(forward, sim_shapes={SIMKEY: (task.dim_data,)})
    store = swyft.MemoryStore(task.dim_parameters, simulator=simulator)

    prior = swyft.Prior.from_uv(prior_transform, task.dim_parameters)
    # theta0_swyft = swyftify_theta(theta0)
    obs0_swyft = swyftify_obs(obs0)

    twod_marginals = list(combinations(range(task.dim_parameters), 2))

    myhps = True
    if myhps:
        train_args = dict(
            batch_size=256,
            validation_size=0.10,
            early_stopping_patience=20,
            max_epochs=300,
            lr=1e-2,
            reduce_lr_factor=0.1,
            reduce_lr_patience=5,
            nworkers=4,
            # optimizer_fn=partial(torch.optim.SGD, momentum=0.9)
        )
        tail_args = dict(
            # hidden_features=128 + 32,
            hidden_features=64,
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

    def custom_new_n(micro):
        n_prev = len(micro._datasets[-1])
        prior = micro._next_priors[-1]
        dataset = swyft.Dataset(
            n_prev, prior, store=micro._store, simhook=micro._simhook
        )
        num_kept = len(dataset)
        return 10_000 + num_kept

    # zoomin
    micro = swyft.Microscope(
        [(i,) for i in range(task.dim_parameters)],
        prior,
        obs0_swyft,
        store=store,
        device=DEVICE,
        Ninit=NINIT,
        new_simulation_factor=1.0,
        convergence_ratio=CONVERGENCE_RATIO,
        epsilon_threshold=EPSILON,
        train_args=train_args,
        tail=get_tail,
        tail_args=tail_args,
        head_args=head_args,
    )
    micro.focus(max_rounds=MAX_ROUNDS, custom_new_n=custom_new_n)
    posterior = swyft.Posteriors.from_Microscope(micro)
    posterior.infer(
        twod_marginals,
        device=DEVICE,
        train_args=train_args,
        tail=get_tail,
        tail_args=tail_args,
        head_args=head_args,
    )

    weighted_samples = posterior.sample(N_POSTERIOR_SAMPLES, obs0_swyft)
    rejection_samples = posterior.rejection_sample(N_POSTERIOR_SAMPLES, obs0_swyft)

    bounds = []
    for ppp in micro.posteriors:
        bbb = ppp._prior.bound
        if isinstance(bbb, swyft.bounds.bounds.UnitCubeBound):
            bound = np.tile([0.0, 1.0], (task.dim_parameters, 1))
        else:
            bound = bbb._bounds[(0, 1, 2)]._rec_bounds
        bounds.append(bound)
    volumes = micro.volumes

    np.save("bounds.npy", bounds)
    tmnre.io.save_array_to_csv("volumes.csv", volumes)
    tmnre.io.save_array_to_csv("n-simulations.csv", micro.n_simulations)
    tmnre.io.save_weighted_samples_to_csv_directory("weighted-samples", weighted_samples)
    tmnre.io.save_marginal_samples_to_csv_directory("rejection-samples", rejection_samples)

    #################################
    ## Metric part
    #################################

    reference_samples = task.get_reference_posterior_samples(NUM_OBS).numpy()
    ref_marginals = tmnre.benchmark.sbibm.load_ref_and_marginalize(task, NUM_OBS)

    def get_kld(samples, ref_samples, limits=(0.0, 1.0), bins=100, add_a_sample=True):
        task_limits = np.ones((task.dim_parameters, 2)) * limits
        return sum(
            tmnre.metrics.kl.kl_summary_swyft(
                swyft_samples=samples,
                ref_samples=ref_samples,
                task_limits=task_limits,
                dim=1,
                bins=bins,
                add_a_sample=add_a_sample,
            ).values()
        )

    kld = get_kld(weighted_samples, reference_samples)
    c2sts = tmnre.metrics.c2st.c2st_on_marginals(
        ref_marginals,
        valmap(get_10k, rejection_samples),
        n_jobs=cfg.hardware.launcher.cpus_per_task,
    )

    tmnre.io.save_array_to_csv("kld.csv", kld)
    tmnre.io.save_dict_arrays_to_csv("c2sts.csv", c2sts)


if __name__ == "__main__":
    main()
