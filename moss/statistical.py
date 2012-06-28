"""Assorted functions for statistical calculations."""
from __future__ import division
import numpy as np
from scipy import stats


def bootstrap(*args, **kwargs):
    """Resample one or more arrays with and calculate a summary statistic.

    Positional arguments are a sequence of arrays to bootrap
    along the first axis and pass to a summary function.

    Keyword arguments:
        n_boot : int
            number of iterations
        func : callable
            function to call on the args that are passed in

    Returns
    -------
    boot_dist: array
        array of bootstrapped statistic values

    """
    # Ensure list of arrays are same length
    if len(np.unique(map(len, args))) > 1:
        raise ValueError("All input arrays must have the same length")
    n = len(args[0])

    # Default keyword arguments
    n_boot = kwargs.get("n_boot", 10000)
    func = kwargs.get("func", np.mean)

    # Do the bootstrap
    boot_dist = []
    for i in xrange(int(n_boot)):
        resampler = np.random.randint(0, n, n)
        sample = [a[resampler] for a in args]
        boot_dist.append(func(*sample))
    return np.array(boot_dist)


def percentiles(a, pcts):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a: array
        data
    pcts: sequence of percentile values
        percentile or percentiles to find score at

    Returns
    -------
    scores: array
        array of scores at requested percentiles

    """
    try:
        scores = np.zeros(len(pcts))
    except TypeError:
        pcts = [pcts]
        scores = np.zeros(1)
    for i, p in enumerate(pcts):
        scores[i] = stats.scoreatpercentile(a, p)
    return scores


def pmf_hist(a, bins=10):
    """Return arguments to plt.bar for pmf-like histogram of an array.

    Parameters
    ----------
    a: array-like
        array to make histogram of
    bins: int
        number of bins

    Returns
    -------
    x: array
        left x position of bars
    h: array
        height of bars
    w: float
        width of bars

    """
    n, x = np.histogram(a, bins)
    h = n / n.sum()
    w = x[1] - x[0]
    return x[:-1], h, w


def add_constant(a):
    """Add a constant term to a design matrix.

    Parameters
    ----------
    a : array
        original design matrix

    Returns
    -------
    a : array
        design matrix with constant as final column

    """
    return np.column_stack((a, np.ones(len(a))))
