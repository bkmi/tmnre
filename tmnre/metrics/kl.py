# pylint: disable=not-callable
from functools import partial
from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.stats import entropy
from toolz import compose, merge, merge_with

from swyft.utils import tensor_to_array
from tmnre.functional.dicttools import filter_by_key_intersection, parallel_merge_with
from tmnre.marginalize import all_limits, all_marginals, filter_marginals_by_dim
from tmnre.metrics.utils import summarize_metric
from tmnre.types import MarginalKey, Marginals


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def one_sample_per_bin(dim: int, limits: np.ndarray, bins: int) -> np.ndarray:
    grid = np.linspace(limits[..., 0], limits[..., 1], bins)
    if dim == 1:
        extra_samples = grid
    elif dim == 2:
        extra_samples = cartesian_product(grid[:, 0], grid[:, 1])
    else:
        raise NotImplementedError("Not implemented for above 2d")
    return extra_samples


def binned_kl(
    ps: np.ndarray,
    qs: np.ndarray,
    limits: np.ndarray,
    bins: int,
    add_a_sample: bool,
    weight_ps: Optional[np.ndarray] = None,
    weight_qs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """kl divergence, D(p || q) = sum(pk * log(pk / qk)), between two arrays of samples via uniform binning.

    Args:
        ps (np.ndarray): samples from p, data distribution
        qs (np.ndarray): samples from q, theory distribution
        limits (np.ndarray): (dim, 2) shape array with the extrema for the bins
        bins (int): number of bins
        add_a_sample (bool): do not allow kl to return nan by adding a sample to every bin
        weight_ps (np.ndarray, optional): Weights for calculating the histogram. Defaults to None.
        weight_qs (np.ndarray, optional): Weights for calculating the histogram. Defaults to None.

    Returns:
        kl divergence (np.ndarray): the kl divergence of the samples, return dim is zero
    """
    assert isinstance(bins, int), "Dimension specific bins is not supported."

    _, d = ps.shape
    assert qs.shape[1] == d
    assert len(limits) == d

    if add_a_sample:
        extra_samples = one_sample_per_bin(d, limits, bins)
        ps = np.concatenate([ps.copy(), extra_samples], axis=0)
        if weight_ps is not None:
            extra_weight_ps = np.min(weight_ps) * np.ones(len(extra_samples))
            weight_ps = np.concatenate([weight_ps.copy(), extra_weight_ps], axis=0)

        qs = np.concatenate([qs.copy(), extra_samples], axis=0)
        if weight_qs is not None:
            extra_weight_qs = np.min(weight_qs) * np.ones(len(extra_samples))
            weight_qs = np.concatenate([weight_qs.copy(), extra_weight_qs], axis=0)

    p, _ = np.histogramdd(ps, bins=bins, range=limits, density=True, weights=weight_ps)
    q, _ = np.histogramdd(qs, bins=bins, range=limits, density=True, weights=weight_qs)
    return entropy(p.flatten(), q.flatten())


def kl_on_marginals(
    X: Marginals,
    Y: Marginals,
    limits: Marginals,
    bins: int,
    add_a_sample: bool = False,
    n_jobs: Optional[int] = None,
    weights_X: Optional[Marginals] = None,
    weights_Y: Optional[Marginals] = None,
) -> Dict[MarginalKey, float]:
    assert isinstance(
        bins, int
    ), "Bins must be uniform across parameters. Alternatives not implemented."
    keys = X.keys()
    assert set(keys) == set(Y.keys())
    for key in keys:
        assert key in set(limits.keys())
    X, Y, limits = filter_by_key_intersection((X, Y, limits))

    assert all(X[k].ndim in [1, 2] for k in keys)
    assert all(Y[k].ndim in [1, 2] for k in keys)
    assert all(limits[k].ndim in [1, 2] for k in keys)

    # Marginals with the same key should have the same dimension
    assert all(X[k].shape[-1] == Y[k].shape[-1] for k in keys)
    assert all(X[k].shape[-1] == len(limits[k]) for k in keys)

    for d in [X, Y, limits]:
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                d[key] = value.numpy()
            else:
                d[key] = np.asarray(value)

    if weights_X is None:
        weights_X = {key: None for key in X.keys()}

    if weights_Y is None:
        weights_Y = {key: None for key in Y.keys()}

    # TODO this could probably be reformulated in a clearer way
    def expand_args_binned_kl(x: List, bins, add_a_sample):
        return binned_kl(
            ps=x[0],
            qs=x[1],
            limits=x[2],
            weight_ps=x[3],
            weight_qs=x[4],
            bins=bins,
            add_a_sample=add_a_sample,
        )

    get_kl = partial(expand_args_binned_kl, bins=bins, add_a_sample=add_a_sample)
    all_kls = parallel_merge_with(
        get_kl,
        [X, Y, limits, weights_X, weights_Y],
        n_jobs=n_jobs,
        inner_max_num_threads=1,
    )
    summary = summarize_metric(all_kls)
    return merge(all_kls, summary)


def kls(
    ps, qs, bins, limits, weight_ps=None, weight_qs=None, add_a_sample=False, dim=None
):
    _, d = ps.shape
    assert qs.shape[1] == d
    assert len(limits) == d

    ps = tensor_to_array(ps)
    qs = tensor_to_array(qs)

    ps_marginals = all_marginals(ps)
    qs_marginals = all_marginals(qs)

    if limits is None:
        limit_marginals = {k: None for k in ps_marginals.keys()}
    else:
        limit_marginals = compose(all_limits, np.asarray)(limits)

    if dim is not None:
        ps_marginals, qs_marginals, limit_marginals = [
            filter_marginals_by_dim(i, dim)
            for i in (ps_marginals, qs_marginals, limit_marginals)
        ]

    # kls from each dict. dict should be [histogram1, histogram2, limits]
    kl_from_dict = merge_with(
        lambda dicts: binned_kl(
            *dicts,
            bins=bins,
            add_a_sample=add_a_sample,
            weight_ps=weight_ps,
            weight_qs=weight_qs
        ),
        [ps_marginals, qs_marginals, limit_marginals],
    )
    return kl_from_dict


# TODO this is more-or-less above but with a dimension restriction
def kl_summary_swyft(
    swyft_samples, ref_samples, task_limits, dim, bins=200, add_a_sample=False
):
    if isinstance(ref_samples, torch.Tensor):
        ref_samples = ref_samples.detach().cpu().numpy()
    ref_marginals = all_marginals(ref_samples)
    ref_marginals_dim = filter_marginals_by_dim(ref_marginals, dim)
    limits_dim = filter_marginals_by_dim(all_limits(task_limits), dim)

    estimated_marginals = all_marginals(swyft_samples["params"])
    estimated_marginals_dim = filter_marginals_by_dim(estimated_marginals, dim)

    estimated_weights_dim = filter_marginals_by_dim(swyft_samples["weights"], dim)
    return kl_on_marginals(
        ref_marginals_dim,
        estimated_marginals_dim,
        limits_dim,
        bins=bins,
        weights_Y=estimated_weights_dim,
        add_a_sample=add_a_sample,
    )


if __name__ == "__main__":
    pass
