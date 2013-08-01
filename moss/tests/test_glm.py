import numpy as np
from scipy import signal

from nose.tools import assert_equal, assert_greater
import numpy.testing as npt

from .. import glm


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
