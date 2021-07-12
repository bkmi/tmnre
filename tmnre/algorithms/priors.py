import swyft
import torch
from swyft.utils.array import tensor_to_array, array_to_tensor
from toolz import compose


def get_affine_uniform_prior(low, high, dim):
    length = high - low
    length = tensor_to_array(length)
    low = tensor_to_array(low)

    def uv(x):
        return length * x + low

    return swyft.Prior.from_uv(uv, dim)


def get_diagonal_normal_prior(loc, precision_matrix):
    d = torch.distributions.normal.Normal(loc, precision_matrix.diagonal())

    assert len(loc) == len(precision_matrix.diagonal())
    dim = len(loc)

    return swyft.Prior(
        ptrans=swyft.bounds.CustomTransform(
            u=compose(tensor_to_array, d.cdf, array_to_tensor),
            v=compose(tensor_to_array, d.icdf, array_to_tensor),
            log_prob=compose(tensor_to_array, d.log_prob, array_to_tensor),
            zdim=dim,
        )
    )
