from __future__ import division

import numpy as np
import scipy as sp
from scipy.stats import gamma


class HRFModel(object):

    def __init__(self):

        raise NotImplementedError

    def __call__(self, timepoints):

        raise NotImplementedError


class GammaDifferenceHRF(HRFModel):

    def __init__(self, pos_shape=4, pos_scale=2,
                 neg_shape=7, neg_scale=2, ratio=.3):
        self.rv_pos = gamma(pos_shape, scale=pos_scale)
        self.rv_neg = gamma(neg_shape, scale=neg_scale)
        self.ratio = ratio

    def __call__(self, timepoints):

        hrf = self.rv_pos.pdf(timepoints)
        hrf -= self.ratio * self.rv_neg.pdf(timepoints)
        hrf /= hrf.sum()
        return hrf


class FIR(HRFModel):

    def __init__(self):

        return NotImplementedError


class DesignMatrix(object):

    pass


def fsl_highpass_matrix(n_tp, cutoff, tr=2):
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


def fsl_highpass_filter(data, cutoff, tr=2, copy=True):
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
        if False data is filtered in place

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
    data[:] = np.dot(F, data)

    return data.squeeze()
