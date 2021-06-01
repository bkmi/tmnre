from typing import Tuple

import numpy as np
import torch

from tmnre.coverage.utils import is_really_1d


def get_credibility_contour_from_histogram(
    histogram_densities, credibility_level: float = 0.68268
) -> np.ndarray:
    if isinstance(histogram_densities, torch.Tensor):
        histogram_densities = histogram_densities.detach().cpu().numpy()

    assert is_really_1d(histogram_densities)
    histogram_densities = np.squeeze(histogram_densities)
    histogram_densities = np.sort(histogram_densities)[::-1]  # Sort backwards
    mass = histogram_densities.sum()
    enclosed_mass = np.cumsum(histogram_densities)
    idx = np.argmax(enclosed_mass >= mass * credibility_level)
    return histogram_densities[idx]


def find_grid_discontinuities(
    almost_grid: np.ndarray,
    interval: float = None,
) -> np.ndarray:
    """If no interval is provided, the most frequent difference is assumed.

    Returns inds of grid discontinuities. Namely, the index is the first gridpoint after the discontinuity.
    """
    assert is_really_1d(almost_grid)
    differences = np.diff(almost_grid)
    if interval is None:
        unique_differences = np.unique(differences)
        frequencies = np.asarray(
            [np.sum(differences == difference) for difference in unique_differences]
        )
        interval = unique_differences[np.argsort(frequencies)][-1]

    jumps = np.logical_and(
        np.greater(differences, interval),
        ~np.isclose(differences, interval),
    )
    jump_inds = np.nonzero(jumps)
    corrected_jump_inds = [jump_ind + 1 for jump_ind in jump_inds]

    if len(corrected_jump_inds) == 1:
        return corrected_jump_inds[0]
    else:
        return corrected_jump_inds


def get_extrema(array: np.ndarray) -> Tuple[np.ndarray]:
    """Return (min, max)"""
    return np.min(array), np.max(array)


def get_credible_intervals(
    samples: np.ndarray,
    weights: np.ndarray = None,
    bins: int = 10,
    credibility_level: float = 0.68268,
):
    """[0.68268, 0.95450, 0.99730]"""
    heights, edges = np.histogram(samples, weights=weights, bins=bins, density=True)
    midpoints = (edges[1:] + edges[:-1]) / 2
    contour = get_credibility_contour_from_histogram(heights, credibility_level)
    region = midpoints[heights >= contour]
    jumps = find_grid_discontinuities(region)
    return np.asarray([get_extrema(s) for s in np.split(region, jumps)])
