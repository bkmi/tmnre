from typing import Dict

import torch
from toolz import compose

from tmnre.marginalize import filter_marginals_by_dim, get_marginal_dim_by_key
from tmnre.types import MarginalKey


def summarize_metric(marginal_metrics: Dict[MarginalKey, float]):
    max_dim = max(get_marginal_dim_by_key(key) for key in marginal_metrics.keys())

    summary = {}
    for dim in range(1, max_dim + 1):
        select_marginals = filter_marginals_by_dim(marginal_metrics, dim)
        data = torch.tensor(list(select_marginals.values()))
        summary[f"{dim}-dim sum"] = compose(torch.atleast_1d, torch.sum)(data)
        summary[f"{dim}-dim mean"] = compose(torch.atleast_1d, torch.mean)(data)
        summary[f"{dim}-dim var"] = compose(torch.atleast_1d, torch.var)(data)
    return summary


if __name__ == "__main__":
    pass
