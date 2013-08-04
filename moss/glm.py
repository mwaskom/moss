from __future__ import division

import numpy as np
import scipy as sp
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt


class HRFModel(object):
    """Abstract class definition for HRF Models."""
    def __init__(self):
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
            dy = np.concatenate([[0], dy])
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

        # Without frametimes, assume data and kernel have same sampling
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
    def __init__(self, design, hrf_model, ntp, regressors=None, confounds=None,
                 artifacts=None, tr=2, hpf_cutoff=128, oversampling=16):
        """Initialize the design matrix object."""
        if "duration" not in design:
            design["duration"] = 0
        if "value" not in design:
            design["value"] = 1

        self.design = design
        self.tr = tr
        frametimes = np.arange(0, ntp * tr, tr, np.float)
        self.frametimes = pd.Series(frametimes, name="frametimes")
        condition_names = np.sort(design.condition.unique())
        self._condition_names = pd.Series(condition_names, name="conditions")
        self._ntp = ntp

        # Convolve the oversampled condition regressors
        self._make_hires_base(oversampling)
        self._convolve(hrf_model)

        # Subsample the condition regressors and highpass filter
        conditions = self._subsample_condition_matrix()
        conditions -= conditions.mean()
        if hpf_cutoff is not None:
            filtered_conditions = self._highpass_filter(conditions, hpf_cutoff)
        else:
            filtered_conditions = conditions

        # Set up the other regressors of interest
        regressors = self._validate_component(regressors, "regressor")

        # Set up the confound submatrix
        confounds = self._validate_component(confounds, "confound")

        # Set up the artifacts submatrix
        if artifacts is not None:
            if artifacts.any():
                n_art = artifacts.sum()
                art = np.zeros((artifacts.size, n_art))
                art[np.where(artifacts), np.arange(n_art)] = 1
                artifacts = self._validate_component(art, "artifact")
            else:
                artifacts = None

        # Now build the full design matrix
        pieces = [filtered_conditions]
        if regressors is not None:
            pieces.append(regressors)
        if confounds is not None:
            pieces.append(confounds)
        if artifacts is not None:
            pieces.append(artifacts)

        X = pd.concat(pieces, axis=1)
        X.index = self.frametimes
        X.columns.name = "evs"
        X -= X.mean(axis=0)
        self.design_matrix = X

        # Now build the column name lists that will let us index
        # into the submatrices
        conf_names, art_names = [], []
        main_names = self._condition_names.tolist()
        if regressors is not None:
            main_names += regressors.columns.tolist()
        if confounds is not None:
            conf_names = confounds.columns.tolist()
        if artifacts is not None:
            art_names = artifacts.columns.tolist()
        self._full_names = X.columns.tolist()
        self._main_names = main_names
        self._confound_names = conf_names
        self._artifact_names = art_names

    def _make_hires_base(self, oversampling):
        """Make the oversampled condition base submatrix."""
        hires_max = self.frametimes.max() * (1 + 1 / (self._ntp - 1))
        hires_ntp = self._ntp * oversampling + 1
        self._hires_ntp = hires_ntp
        self._hires_frametimes = np.linspace(0, hires_max, hires_ntp)

        hires_base = pd.DataFrame(columns=self._condition_names,
                                  index=self._hires_frametimes)

        for cond in self._condition_names:
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
        self._hires_X = self._hires_base.copy()
        for cond in self._condition_names:
            res = hrf_model.convolve(self._hires_base[cond],
                                     self._hires_frametimes)
            for key, vals in res.iteritems():
                self._hires_X[key] = vals

    def _subsample_condition_matrix(self):
        """Sample the hires convolved matrix at the TR midpoint."""
        condition_X = pd.DataFrame(columns=self._hires_X.columns,
                                   index=self.frametimes)

        frametime_midpoints = self.frametimes + self.tr / 2
        for key, vals in self._hires_X.iteritems():
            resampler = sp.interpolate.interp1d(self._hires_frametimes, vals)
            condition_X[key] = resampler(frametime_midpoints)

        return condition_X

    def _validate_component(self, comp, name_base):
        """For components that can be an an array or df, build the df."""
        if comp is None:
            return None
        try:
            names = comp.columns
        except AttributeError:
            n = comp.shape[1]
            names = pd.Series([name_base + "_%d"] * n) % range(n)
            comp = pd.DataFrame(comp, self.frametimes, names)

        if not np.all(comp.index == self.frametimes):
            err = "Frametimes for %ss do not match design." % name_base
            raise ValueError(err)

        return comp

    def _highpass_filter(self, mat, cutoff):
        """Highpass-filter each column in mat."""
        F = fsl_highpass_matrix(self._ntp, cutoff, self.tr)
        for key, vals in mat.iteritems():
            mat[key] = np.dot(F, vals)
        return mat

    def contrast_vector(self, cols, contrast):
        """Return a full contrast vector given conditions and weightings."""
        vector = np.zeros(self.design_matrix.shape[1])
        for name, val in zip(cols, contrast):
            vector[self.design_matrix.columns == name] = val
        return vector

    def plot(self, kind="main", fname=None, cmap="bone"):
        """Draw an image representation of the design matrxi."""
        names = getattr(self, "_%s_names" % kind)
        mat = self.design_matrix[names]
        mat = mat / mat.abs().max()

        x, y = .66 * mat.shape[1], .02 * mat.shape[0]
        figsize = min(x, 10), min(y, 14)
        f, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                  interpolation="nearest", zorder=2)
        ax.set_yticks([])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, ha="right", rotation=30)
        for x in range(len(names) - 1):
            ax.axvline(x + .5, c="#222222", lw=3, zorder=3)
        plt.tight_layout()

        if fname is not None:
            f.savefig(fname)

    def to_csv(self, fname):
        """Save the full design matrix to csv."""
        self.design_matrix.to_csv(fname)

    def to_fsl_files(self, fname, contrasts=None):
        """Save to FEAT-style .mat and (optionally) .con files."""
        pass

    @property
    def main_submatrix(self):
        """Conditions (no derivatives) and regressors."""
        return self.design_matrix[self._main_names]

    @property
    def condition_submatrix(self):
        """Only condition information."""
        return self.design_matrix[self._condition_names]

    @property
    def confound_submatrix(self):
        """Submatrix of confound regressors."""
        if not self._confound_names:
            return None
        return self.design_matrix[self._confound_names]

    @property
    def artifact_submatrix(self):
        """Submatrix of artifact regressors."""
        if not self._artifact_names:
            return None
        return self.design_matrix[self._artifact_names]


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
