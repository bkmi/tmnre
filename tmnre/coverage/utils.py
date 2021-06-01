import numpy as np
import torch


def is_really_1d(x) -> bool:
    if isinstance(x, np.ndarray):
        squeezed = np.squeeze(x)
    elif isinstance(x, torch.Tensor):
        squeezed = torch.squeeze(x)
    else:
        raise NotImplementedError
    return squeezed.ndim == 1
