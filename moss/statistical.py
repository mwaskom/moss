"""Assorted functions for statistical calculations."""
from __future__ import division
import numpy as np
import scipy as sp
from scipy import stats


def bootstrap(*args, **kwargs):
    """Resample one or more arrays and call a function on each sample.

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


def percentiles(a, pcts, axis=None):
    """Like scoreatpercentile but can take and return array of percentiles.

    Parameters
    ----------
    a : array
        data
    pcts : sequence of percentile values
        percentile or percentiles to find score at
    axis : int or None
        if not None, computes scores over this axis

    Returns
    -------
    scores: array
        array of scores at requested percentiles
        first dimension is length of object passed to ``pcts``

    """
    scores = []
    try:
        n = len(pcts)
    except TypeError:
        pcts = [pcts]
        n = 0
    for i, p in enumerate(pcts):
        if axis is None:
            score = stats.scoreatpercentile(a.ravel(), p)
        else:
            score = np.apply_along_axis(stats.scoreatpercentile, axis, a, p)
        scores.append(score)
    scores = np.asarray(scores)
    if not n:
        scores = scores.squeeze()
    return scores


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


def fsl_highpass_matrix(n_tp, cutoff, tr=1):
    """Return a matrix to implement FSL's gaussian running line filter.

    This returns a matrix that you premultiply your data with to
    implement the filter.

    Parameters
    ----------
    n_tp : int
        number of observations in data
    cutoff : float
        filter cutoff in seconds
    tr : float
        TR of data in seconds

    Return
    ------
    F : n_tp square array
        filter matrix

    """
    cutoff = cutoff / tr
    sig2n = np.square(cutoff / np.sqrt(2))

    kernel = np.exp(-np.square(np.arange(n_tp)) / (2 * sig2n))
    kernel = 1 / np.sqrt(2 * np.pi * sig2n) * kernel

    K = sp.linalg.toeplitz(kernel)
    K = np.dot(np.diag(1 / K.sum(axis=1)), K)

    H = np.zeros((n_tp, n_tp))
    X = np.column_stack((np.ones(n_tp), np.arange(n_tp)))
    for k in xrange(n_tp):
        W = np.diag(K[k])
        hat = np.dot(np.dot(X, np.linalg.pinv(np.dot(W, X))), W)
        H[k] = hat[k]
    F = np.eye(n_tp) - H
    return F


def fsl_highpass_filter(data, cutoff, tr=1, copy=True):
    """Highpass filter data with gaussian running line filter.

    Parameters
    ----------
    data : 1d or 2d array
        data array where first dimension is observations
    cutoff : float
        filter cutoff in seconds
    tr : float
        data TR in seconds
    copy : boolean
        if True makes copy of data before filtering

    Returns
    -------
    data : 1d or 2d array
        filtered version of the data

    """
    if copy:
        data = data.copy()
    # Ensure data is in right shape
    n_tp = len(data)
    data = np.atleast_2d(data).reshape(n_tp, -1)

    # Filter each column of the data
    F = fsl_highpass_matrix(n_tp, cutoff, tr)
    for j, col in enumerate(data.T):
        data[:, j] = np.dot(F, col)

    return data.squeeze()


def gamma_params(peak, fwhm):
    """Return parameters to scipy.stats.gamma corresponding for an HRF shape.

    This was mostly copied from nipy.

    Parameters
    ----------
    peak : float
        time of response peak
    fwhm : float
        fwhm of response curve

    Returns
    -------
    shape : float
        shape parameter for gamma dist
    scale : float
        scale parameter for gamma dist (1 / lambda)

    """
    shape = np.power(peak / fwhm, 2) * 8 * np.log(2.0)
    scale = np.power(fwhm, 2) / peak / 8 / np.log(2.0)
    return shape, scale


def gamma_hrf(x, peak=5.4, fwhm=5.2):
    """Evaluate a single gamma HRF at given timepoints.

    Parameters
    ----------
    x : scalar or array
        time (in seconds) to evaluate HRF at
    peak : float
        time of response peak
    fwhm : float
        fwhm of response curve

    Returns
    -------
    hrf : scalar or array
        response height at given timepoints

    """
    shape, scale = gamma_params(peak, fwhm)
    return stats.gamma(shape, scale=scale).pdf(x)


def dgamma_hrf(x, r_peak=5.4, r_fwhm=5.2, u_peak=10.8, u_fwhm=7.35, ratio=.35):
    """Evaluate a difference of gammas HRF at given timepoints.

    Default params come from Glover 1999.

    Parameters
    ----------
    x : scalar or array
        time (in seconds) to evaluate HRF at
    r_peak : float
        time of response peak
    r_fwhm : float
        fwhm of response curve
    u_peak : float
        time of undershoot peak
    u_fwhm : float
        fwhm of undershoot curve
    ratio : float
        ratio of response to undershoot

    Returns
    -------
    hrf : scalar or array
        response height at given timepoints

    """
    r_shape, r_scale = gamma_params(r_peak, r_fwhm)
    r_gamma = stats.gamma(r_shape, scale=r_scale)
    u_shape, u_scale = gamma_params(u_peak, u_fwhm)
    u_gamma = stats.gamma(u_shape, scale=u_scale)
    hrf_func = lambda x: r_gamma.pdf(x) - u_gamma.pdf(x) * ratio
    constant, err = sp.integrate.quad(hrf_func, 0, np.inf)
    return hrf_func(x) / constant
