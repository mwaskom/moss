import numpy as np
import pandas as pd
from scipy import signal

import nose.tools as nt
from nose.tools import assert_equal, assert_greater
import numpy.testing as npt

from .. import glm


def test_hrf_sum():
    """Returned HRF values should sum to 1."""
    hrf1 = glm.GammaDifferenceHRF()
    npt.assert_almost_equal(hrf1.kernel.sum(), 1)

    hrf2 = glm.GammaDifferenceHRF(ratio=0)
    npt.assert_almost_equal(hrf2.kernel.sum(), 1)


def test_hrf_peaks():
    """Test HRF based on gamma distribution properties."""
    hrf1 = glm.GammaDifferenceHRF(oversampling=500,
                                  pos_shape=6, pos_scale=1, ratio=0)
    hrf1_peak = hrf1._timepoints[np.argmax(hrf1.kernel)]
    npt.assert_almost_equal(hrf1_peak, 5, 2)

    hrf2 = glm.GammaDifferenceHRF(oversampling=500,
                                  pos_shape=4, pos_scale=2, ratio=0)
    hrf2_peak = hrf2._timepoints[np.argmax(hrf2.kernel)]
    npt.assert_almost_equal(hrf2_peak, (4 - 1) * 2, 2)

    hrf3 = glm.GammaDifferenceHRF(oversampling=500,
                                  neg_shape=7, neg_scale=2, ratio=1000)
    hrf3_trough = hrf3._timepoints[np.argmax(hrf3.kernel)]
    npt.assert_almost_equal(hrf3_trough, (7 - 1) * 2, 2)


def test_hrf_shape():
    """Test the shape of the hrf output with different params."""
    hrf1 = glm.GammaDifferenceHRF()
    npt.assert_equal(hrf1.kernel.shape[1], 1)

    hrf2 = glm.GammaDifferenceHRF(temporal_deriv=True)
    npt.assert_equal(hrf2.kernel.shape[1], 2)


def test_hrf_deriv_scaling():
    """Test relative scaling of main HRF and derivative."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    y, dy = hrf.kernel.T
    ss_y = np.square(y).sum()
    ss_dy = np.square(dy).sum()
    npt.assert_almost_equal(ss_y, ss_dy)


def test_hrf_deriv_timing():
    """Gamma derivative should peak earlier than the main HRF."""
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)
    y, dy = hrf.kernel.T
    nt.assert_greater(np.argmax(y), np.argmax(dy))


def test_hrf_convolution():
    """Test some basics about the convolution."""
    hrf = glm.GammaDifferenceHRF()
    data1 = np.zeros(500)
    data1[0] = 1
    conv1 = hrf.convolve(data1)
    npt.assert_almost_equal(conv1.sum(), 1)

    data2 = np.ones(500)
    conv2 = hrf.convolve(data2)
    npt.assert_almost_equal(conv2.ix[-200:].mean(), 1)


def test_hrf_frametimes():
    """Test the frametimes that come out of the convolution."""
    data = (np.random.rand(500) < .2).astype(int)
    ft = np.arange(500)

    hrf1 = glm.GammaDifferenceHRF()
    conv1 = hrf1.convolve(data, ft)
    npt.assert_array_equal(ft, conv1.index.values)

    hrf2 = glm.GammaDifferenceHRF(tr=1, oversampling=1)
    conv2 = hrf2.convolve(data)
    npt.assert_array_equal(ft, conv2.index.values)

    hrf3 = glm.GammaDifferenceHRF(tr=2, oversampling=2)
    conv3 = hrf3.convolve(data)
    npt.assert_array_equal(ft, conv3.index.values)


def test_hrf_names():
    """Test the names that come out of the convolution."""
    data = (np.random.rand(500) < .2).astype(int)
    series_data = pd.Series(data, name="donna")

    hrf1 = glm.GammaDifferenceHRF()

    conv1 = hrf1.convolve(data)
    nt.assert_equal(conv1.columns.tolist(), ["event"])

    conv2 = hrf1.convolve(data, name="donna")
    nt.assert_equal(conv2.columns.tolist(), ["donna"])

    conv3 = hrf1.convolve(series_data)
    nt.assert_equal(conv3.columns.tolist(), ["donna"])

    hrf2 = glm.GammaDifferenceHRF(temporal_deriv=True)
    conv4 = hrf2.convolve(series_data)
    nt.assert_equal(conv4.columns.tolist(), ["donna", "donna_deriv"])


def test_highpass_matrix_shape():
    """Test the filter matrix is the right shape."""
    for n_tp in 10, 100:
        F = glm.fsl_highpass_matrix(n_tp, 50)
        assert_equal(F.shape, (n_tp, n_tp))


def test_filter_matrix_diagonal():
    """Test that the filter matrix has strong diagonal."""
    F = glm.fsl_highpass_matrix(10, 3)
    npt.assert_array_equal(F.argmax(axis=1).squeeze(), np.arange(10))


def test_filtered_data_shape():
    """Test that filtering data returns same shape."""
    data = np.random.randn(100)
    data_filt = glm.fsl_highpass_filter(data, 30)
    assert_equal(data.shape, data_filt.shape)

    data = np.random.randn(100, 3)
    data_filt = glm.fsl_highpass_filter(data, 30)
    assert_equal(data.shape, data_filt.shape)


def test_filter_psd():
    """Test highpass filter with power spectral density."""
    a = np.sin(np.linspace(0, 4 * np.pi, 256))
    b = np.random.randn(256) / 2
    y = a + b
    y_filt = glm.fsl_highpass_filter(y, 10)
    assert_equal(y.shape, y_filt.shape)

    _, orig_d = signal.welch(y)
    _, filt_d = signal.welch(y_filt)

    assert_greater(orig_d.sum(), filt_d.sum())


def test_filter_strength():
    """Test that lower cutoff makes filter more aggresive."""
    a = np.sin(np.linspace(0, 4 * np.pi, 256))
    b = np.random.randn(256) / 2
    y = a + b

    cutoffs = np.linspace(20, 80, 5)
    densities = np.zeros_like(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        filt = glm.fsl_highpass_filter(y, cutoff)
        _, density = signal.welch(filt)
        densities[i] = density.sum()

    npt.assert_array_equal(densities, np.sort(densities))


def test_filter_copy():
    """Test that copy argument to filter function works."""
    a = np.random.randn(100, 10)
    a_copy = glm.fsl_highpass_filter(a, 50, copy=True)
    assert(not (a == a_copy).all())
    a_nocopy = glm.fsl_highpass_filter(a, 100, copy=False)
    npt.assert_array_equal(a, a_nocopy)
