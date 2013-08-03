from __future__ import division

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import gamma


class HRFModel(object):
    """Abstract class definition for HRF Models."""
    def __init__(self):
        raise NotImplementedError

    def __call__(self, timepoints):
        """Evaluate the HRF at timepoints."""
        raise NotImplementedError

    @property
    def kernel(self):
        """Evaluate the kernal at timepoints."""
        raise NotImplementedError

    def convolve(self, data):
        """Convolve the kernel with some data."""
        raise NotImplementedError


class GammaDifferenceHRF(HRFModel):
    """Canonical difference of gamma variates HRF model."""
    def __init__(self, tr=2, oversampling=16, temporal_deriv=False,
                 kernel_secs=32, pos_shape=4, pos_scale=2,
                 neg_shape=7, neg_scale=2, ratio=.3):
        """Create the HRF object with Glover parameters as default."""
        self._rv_pos = gamma(pos_shape, scale=pos_scale)
        self._rv_neg = gamma(neg_shape, scale=neg_scale)
        self._tr = tr
        self._oversampling = oversampling
        dt = tr / oversampling
        self._timepoints = np.linspace(0, kernel_secs, kernel_secs / dt)
        self._temporal_deriv = temporal_deriv
        self._ratio = ratio

    @property
    def kernel(self):
        """Evaluate the kernel at timepoints, maybe with derivative."""
        y = self._rv_pos.pdf(self._timepoints)
        y -= self._ratio * self._rv_neg.pdf(self._timepoints)
        y /= y.sum()

        if self._temporal_deriv:
            dy = np.diff(y)
            dy = np.concatenate([dy, [0]])
            scale = np.sqrt(np.square(y).sum() / np.square(dy).sum())
            dy *= scale
            y = np.c_[y, dy]
        else:
            y = np.c_[y]

        return y

    def convolve(self, data, frametimes=None, name=None):
        """Convolve the kernel with some data.

        Parameters
        ----------
        data : Series or 1d array
            data to convolve
        frametimes : Series or 1d array, optional
            timepoints corresponding to data - if None, assume
            data is sampled with same TR and oversampling as kernal
        name : string
            name to associate with data if not passing Series object

        Returns
        -------
        out : DataFrame
            n_cols depends on whether kernel has temporal derivative

        """
        ntp = len(data)
        if frametimes is None:
            orig_ntp = (ntp - 1) / self._oversampling
            orig_max = (orig_ntp - 1) * self._tr
            this_max = orig_max * (1 + 1 / (orig_ntp - 1))
            frametimes = np.linspace(0, this_max, ntp)

        # Get the output name for this condition
        if name is None:
            try:
                name = data.name
            except AttributeError:
                name = "event"
        cols = [name]

        # Obtain the current kernel and set up the output
        kernel = self.kernel.T
        out = np.empty((ntp, len(kernel)))

        # Do the convolution
        if self._temporal_deriv:
            cols.append(name + "_deriv")
            main, deriv = kernel
            out[:, 0] = np.convolve(data, main)[:ntp]
            out[:, 1] = np.convolve(data, deriv)[:ntp]
        else:
            out[:, 0] = np.convolve(data, kernel.ravel())[:ntp]

        # Build the output DataFrame
        out = pd.DataFrame(out, columns=cols, index=frametimes)
        return out


class FIR(HRFModel):
    """Finite Impule Response HRF model."""
    def __init__(self):

        return NotImplementedError


class DesignMatrix(object):
    """fMRI-specific design matrix object."""
    def __init__(self, design, condition_names, hrf_model, tr, ntp,
                 oversampling=16):
        """Initialize the design matrix object."""
        if "duration" not in design:
            design["duration"] = 0
        if "value" not in design:
            design["value"] = 0

        self.design = design
        self.condition_names = condition_names
        self.tr = tr
        self.ntp = ntp
        self.frametimes = np.arange(0, (ntp * tr) - 1, tr, np.float)

        # Make an oversampled version of the condition base submatrix
        self._make_hires_base(oversampling)
        self._convolve(hrf_model)

    def _make_hires_base(self, oversampling):
        """Make the oversampled condition base submatrix."""
        hires_max = self.frametimes.max() * (1 + 1 / (self.ntp - 1))
        hires_ntp = self.ntp * oversampling + 1
        self._hires_ntp = hires_ntp
        self._hires_frametimes = np.linspace(0, hires_max, hires_ntp)

        hires_base = pd.DataFrame(columns=self.condition_names,
                                  index=self._hires_frametimes)

        for cond in self.condition_names:
            cond_info = self.design[self.design.condition == cond]
            cond_info = cond_info[["onset", "duration", "value"]]
            regressor = self._make_hires_regressor_base(cond_info)
            hires_base[cond] = regressor
        self._hires_base = hires_base

    def _make_hires_regressor_base(self, info):
        """Oversample a condition regressor."""
        hft = self._hires_frametimes

        # Get the condition information
        onsets, durations, vals = info.values.T

        # Make the regressor timecourse
        tmax = len(hft)
        regressor = np.zeros_like(hft).astype(np.float)
        t_onset = np.minimum(np.searchsorted(hft, onsets), tmax - 1)
        regressor[t_onset] += vals
        t_offset = np.minimum(np.searchsorted(hft, onsets + durations),
                              len(hft) - 1)

        # Handle the case where duration is 0 by offsetting at t + 1
        for i, off in enumerate(t_offset):
            if off < (tmax - 1) and off == t_onset[i]:
                t_offset[i] += 1

        regressor[t_offset] -= vals
        regressor = np.cumsum(regressor)

        return regressor

    def _convolve(self, hrf_model):
        """Convolve the condition regressors with the HRF model."""
        pass


def fsl_highpass_matrix(n_tp, cutoff, tr=2):
    """Return an array to implement FSL's gaussian running line filter.

    To implement the filter, premultiply your data with this array.

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


def fsl_highpass_filter(data, cutoff=128, tr=2, copy=True):
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
