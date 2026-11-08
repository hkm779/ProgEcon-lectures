"""Simulation tools for the Danish wealth inequality project."""

import numpy as np


def _validate_age_shares(age_shares: np.ndarray) -> np.ndarray:
    """Return valid age shares as a NumPy array."""
    shares = np.asarray(age_shares, dtype=float)
    if shares.ndim != 1 or shares.size == 0:
        raise ValueError("age_shares must be a non-empty one-dimensional array")
    if np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("age_shares must be non-negative and sum to one")
    return shares


def _draw_ages(rng: np.random.Generator, age_shares: np.ndarray, n: int) -> np.ndarray:
    """Draw age-group indices by assigning uniform draws to share-sized intervals."""
    uniforms = rng.uniform(0.0, 1.0, size=n)
    age = np.full(n, -1, dtype=int)
    lower = 0.0

    for group, share in enumerate(age_shares):
        upper = lower + share
        if group == len(age_shares) - 1:
            upper = 1.0  # include any floating-point remainder in the final interval
        age[(uniforms >= lower) & (uniforms < upper)] = group
        lower = upper

    if np.any(age < 0):
        raise RuntimeError("Some simulated adults were not assigned an age group")
    return age


def simulate_median_economy(
    age_shares: np.ndarray,
    age_medians: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate adults who all receive their age group's median wealth.

    Parameters
    ----------
    age_shares : array-like
        Population shares in the same order as ``age_medians``.
    age_medians : array-like
        Published P50 wealth for each age group, in DKK.
    n : int, default 10_000
        Number of adults to simulate.
    seed : int, default 42
        Seed for NumPy's random number generator.

    Returns
    -------
    age : numpy.ndarray
        Simulated age-group indices from zero to the number of groups minus one.
    wealth : numpy.ndarray
        Simulated net wealth in DKK.
    """
    shares = _validate_age_shares(age_shares)
    medians = np.asarray(age_medians, dtype=float)
    if medians.ndim != 1 or medians.size != shares.size:
        raise ValueError("age_medians must have one value for each age share")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    age = _draw_ages(rng, shares, n)
    wealth = medians[age]
    return age, wealth


def simulate_percentile_economy(
    age_shares: np.ndarray,
    percentile_values: np.ndarray,
    n: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate wealth varying within age groups using published P1--P99 values.

    Each adult receives an independently drawn rank from 1 through 99, with all
    ranks equally likely, and the corresponding threshold for their age group.

    Parameters
    ----------
    age_shares : array-like
        Population shares in the same order as the rows of ``percentile_values``.
    percentile_values : array-like
        Two-dimensional array with age groups in rows and P1--P99 in columns.
    n : int, default 10_000
        Number of adults to simulate.
    seed : int, default 42
        Seed for NumPy's random number generator.

    Returns
    -------
    age : numpy.ndarray
        Simulated age-group indices from zero to the number of groups minus one.
    wealth : numpy.ndarray
        Simulated net wealth in DKK.
    """
    shares = _validate_age_shares(age_shares)
    values = np.asarray(percentile_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != shares.size or values.shape[1] != 99:
        raise ValueError("percentile_values must have one row per age group and 99 columns")
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rng = np.random.default_rng(seed)
    age = _draw_ages(rng, shares, n)
    ranks = rng.integers(1, 100, size=n)
    wealth = values[age, ranks - 1]
    return age, wealth
