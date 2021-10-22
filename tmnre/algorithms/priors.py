import torch
from toolz import compose

import swyft
from swyft.utils.array import array_to_tensor, tensor_to_array


def get_affine_uniform_prior(low, high, dim):
    length = high - low
    length = tensor_to_array(length)
    low = tensor_to_array(low)

    def uv(x):
        return length * x + low

    return swyft.Prior.from_uv(uv, dim)


def get_diagonal_normal_prior(loc: torch.Tensor, precision_matrix: torch.Tensor):
    d = torch.distributions.normal.Normal(loc, precision_matrix.diagonal())

    assert len(loc) == len(precision_matrix.diagonal())
    dim = len(loc)

    # return swyft.Prior.from_uv(compose(tensor_to_array, d.cdf, array_to_tensor), dim)
    return swyft.Prior(
        ptrans=swyft.bounds.CustomTransform(
            u=compose(tensor_to_array, d.cdf, array_to_tensor),
            v=compose(tensor_to_array, d.icdf, array_to_tensor),
            log_prob=compose(tensor_to_array, d.log_prob, array_to_tensor),
            zdim=dim,
        )
    )


def get_diagonal_lognormal_prior(loc: torch.Tensor, scale: torch.Tensor):
    assert loc.ndim == 1
    assert scale.ndim == 1
    assert len(loc) == len(scale)
    dim = len(loc)
    d = torch.distributions.LogNormal(loc=loc, scale=scale)

    # ###### for scipy.stats
    # # A common parametrization for a lognormal random variable Y is in terms of the mean, mu, and standard deviation, sigma,
    # # of the unique normally distributed random variable X such that exp(X) = Y.
    # # This parametrization corresponds to setting s = sigma and scale = exp(mu).
    # sds = [
    #     stats.lognorm(s=sigma, loc=1, scale=exp(mu))
    #     for mu, sigma in zip(loc.tolist(), scale.tolist())
    # ]

    return swyft.Prior(
        ptrans=swyft.bounds.CustomTransform(
            u=compose(tensor_to_array, d.cdf, array_to_tensor),
            v=compose(tensor_to_array, d.icdf, array_to_tensor),
            log_prob=compose(tensor_to_array, d.log_prob, array_to_tensor),
            zdim=dim,
        )
    )
