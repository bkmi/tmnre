# pylint: disable=not-callable
from functools import partial
from typing import Dict, Optional

import torch
from joblib import Parallel, delayed, parallel_backend
from toolz import merge, merge_with, valmap

from sbibm.metrics import c2st
from swyft.types import Array
from swyft.utils import array_to_tensor
from tmnre.functional.dicttools import map_on_value_pairs
from tmnre.marginalize import all_marginals
from tmnre.metrics.utils import summarize_metric


def c2sts(ps, qs, n_jobs=None):
    ps_marginals = all_marginals(ps)
    qs_marginals = all_marginals(qs)
    if n_jobs is None:
        return merge_with(
            lambda x: float(c2st(*map(torch.tensor, x))), [ps_marginals, qs_marginals]
        )
    else:
        assert ps_marginals.keys() == qs_marginals.keys()
        fn = lambda x, y: float(c2st(torch.tensor(x), torch.tensor(y)))  # noqa: E731
        with parallel_backend("loky", inner_max_num_threads=1):
            result = Parallel(n_jobs=n_jobs)(
                delayed(fn)(p, q)
                for p, q in zip(ps_marginals.values(), qs_marginals.values())
            )
        return {k: r for k, r in zip(ps_marginals.keys(), result)}


def c2st_on_marginals(
    X: Dict[str, Array],
    Y: Dict[str, Array],
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    get_float_c2st = partial(
        c2st,
        seed=seed,
        n_folds=n_folds,
        scoring=scoring,
        z_score=z_score,
        noise_scale=noise_scale,
    )
    assert set(X.keys()) == set(Y.keys())
    assert all(X[k].shape == Y[k].shape for k in X.keys())
    x = valmap(array_to_tensor, X)
    y = valmap(array_to_tensor, Y)
    all_c2sts = map_on_value_pairs(
        get_float_c2st,
        x,
        y,
        backend="loky",
        n_jobs=n_jobs,
        inner_max_num_threads=1,
    )
    # summary = summarize_by_dim_using_comma_count(all_c2sts)
    summary = summarize_metric(all_c2sts)
    return merge(all_c2sts, summary)


if __name__ == "__main__":
    pass
