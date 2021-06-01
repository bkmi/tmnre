from typing import Callable, Collection, Dict, Mapping, Optional, Tuple, TypeVar

from joblib import Parallel, delayed, parallel_backend
from toolz import keyfilter, merge_with

A = TypeVar("A")
B = TypeVar("B")


def map_on_value_pairs(
    func: Callable[[A, A], B],
    x: Dict[str, A],
    y: Dict[str, A],
    backend: Optional[str] = "loky",
    n_jobs: Optional[int] = None,
    inner_max_num_threads: Optional[int] = None,
) -> Dict[str, B]:
    """Use a pairwise function on matching dictionary entries.

    Args:
        func (Callable[[A, A], B]): pariwise function
        x (Dict[str, A]): first input to function, keywise
        y (Dict[str, A]): second input to function, keywise
        backend (Optional[str]): joblib backend. Defaults to "loky".
        n_jobs (Optional[int]): n_jobs, None means do-not-parallelize, joblib.Parallel n_jobs otherwise. Defaults to None.
        inner_max_num_threads (Optional[int]): maximum threads on lower level calls. Defaults to None.

    Returns:
        Dict[str, B]: pairwise function applied to each pair of keywise values in the input dictionaries

    Reference:
        joblib.parallel_backend
    """
    assert set(x.keys()) == set(y.keys())
    keys = x.keys()

    if n_jobs is None:
        return {k: func(x[k], y[k]) for k in keys}
    else:
        with parallel_backend(backend, inner_max_num_threads):
            gen = (delayed(func)(x[k], y[k]) for k in keys)
            result = Parallel(n_jobs=n_jobs)(gen)
        return {k: r for k, r in zip(keys, result)}


def parallel_merge_with(
    func: Callable,
    dicts: Collection[Dict],
    backend: Optional[str] = "loky",
    n_jobs: Optional[int] = None,
    inner_max_num_threads: Optional[int] = None,
) -> Dict:
    """Use a pairwise function on matching dictionary entries.

    Args:
        func (Callable): function that takes a list
        dicts (Collection[Dict]): each value is given to function in a list, in order of dicts
        backend (Optional[str]): joblib backend. Defaults to "loky".
        n_jobs (Optional[int]): n_jobs, None means do-not-parallelize, joblib.Parallel n_jobs otherwise. Defaults to None.
        inner_max_num_threads (Optional[int]): maximum threads on lower level calls. Defaults to None.

    Returns:
        Dict: pairwise function applied to each pair of keywise values in the input dictionaries

    Reference:
        joblib.parallel_backend
        toolz.dicttoolz.merge_with
    """
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]

    if n_jobs is None:
        return merge_with(func, *dicts)
    else:
        _result = {}
        for d in dicts:
            for k, v in d.items():
                if k not in _result:
                    _result[k] = [v]
                else:
                    _result[k].append(v)
        with parallel_backend(backend, inner_max_num_threads):
            gen = (delayed(func)(value) for value in _result.values())
            result = Parallel(n_jobs=n_jobs)(gen)
        return {k: r for k, r in zip(d.keys(), result)}


def filter_by_key_intersection(*dicts: Collection[dict]) -> Tuple[dict]:
    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]

    sets_of_keys = [set(d.keys()) for d in dicts]
    intersection = set.intersection(*sets_of_keys)
    is_in_intersection = lambda x: x in intersection
    return [keyfilter(is_in_intersection, d) for d in dicts]
