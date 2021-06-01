from itertools import combinations
from typing import Sequence, Union

import numpy as np
from toolz.dicttoolz import keyfilter

from tmnre.types import Marginals
from swyft.types import Array


def corner_combinations(zdim: int):
    """The pairwise combinations required for creating a corner plot.

    Args:
        zdim

    Returns:
        pairwise combinations, i.e. [[0,1], [0,2], [1,2]]
    """
    return combinations(range(zdim), 2)


def names_to_sorted_keys(param_names: Union[np.ndarray, str]):
    if isinstance(param_names, np.ndarray):
        param_names = param_names.tolist()
    elif isinstance(param_names, str):
        param_names = param_names.split(",")
    else:
        raise TypeError("Try wrapping lists / sequences with np.asarray(input)")
    param_names = sorted(param_names)
    return ",".join([str(name) for name in param_names])


def all_indices(dim, param_names=None):
    if param_names is None:

        def get_name(x):
            if isinstance(x, int):
                return (x,)
            else:
                return tuple(x)

    else:

        def get_name(x):
            return names_to_sorted_keys(param_names[x])

        param_names = np.asarray(param_names)
    to_marginalize = {get_name(i): [i] for i in range(dim)}
    to_marginalize.update({get_name(i): i for i in corner_combinations(dim)})
    return to_marginalize


def _select_limit(limits: Array, fancy_inds: Union[int, Sequence]) -> Array:
    if isinstance(fancy_inds, int):
        fancy_inds = [fancy_inds]
    else:
        fancy_inds = list(fancy_inds)
    return limits[fancy_inds, ...]


def all_limits(limits, param_names=None) -> Marginals:
    """get all corner combinations of limits

    Args:
        limits: [total_dim, 2]
        param_names: output these names as keys

    Returns:
        limit dictionary
    """
    limits = np.asarray(limits)
    assert limits.ndim == 2
    dim = limits.shape[0]
    to_marginalize = all_indices(dim, param_names)
    return {k: _select_limit(limits, v) for k, v in to_marginalize.items()}


def marginalize(array, fancy_inds):
    if isinstance(fancy_inds, int):
        fancy_inds = [fancy_inds]
    else:
        fancy_inds = list(fancy_inds)
    return array[..., fancy_inds]


def all_marginals(array, param_names=None) -> Marginals:
    assert array.ndim == 2
    dim = array.shape[1]
    to_marginalize = all_indices(dim, param_names)
    return {k: marginalize(array, v) for k, v in to_marginalize.items()}


def get_marginal_dim_by_key(key: tuple) -> int:
    return len(key)


def get_marginal_dim_by_value(value: Array) -> int:
    return value.shape[-1]


def filter_marginals_by_dim(marginals: Marginals, dim: int) -> Marginals:
    assert all(
        isinstance(k, tuple) for k in marginals.keys()
    ), "This function works on tuples of parameters."
    return keyfilter(lambda x: get_marginal_dim_by_key(x) == dim, marginals)


check_if_x_dim = {
    "1-dim": lambda x: get_marginal_dim_by_key(x) == 1,
    "2-dim": lambda x: get_marginal_dim_by_key(x) == 2,
}  # This should be called sort into dims or something.


if __name__ == "__main__":
    names = ["a", "b", "c"]
    dim = len(names)
    limits = [[-1, 1], [-2.5, 0.5], [-0.2, 100]]
    print(all_indices(dim, names))
    print(all_limits(limits, names))
    print(all_marginals(np.random.rand(1, dim), names))
