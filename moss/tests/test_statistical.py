import numpy as np
import scipy as sp
from matplotlib.mlab import psd

from numpy.testing import assert_array_equal, assert_array_almost_equal
import nose.tools
from nose.tools import assert_equal, raises

from .. import statistical as stat


a_norm = np.random.randn(100)

a_range = np.arange(101)


def test_bootstrap():
    """Test that bootstrapping gives the right answer in dumb cases."""
    a_ones = np.ones(10)
    n_boot = 5
    out1 = stat.bootstrap(a_ones, n_boot=n_boot)
    assert_array_equal(out1, np.ones(n_boot))
    out2 = stat.bootstrap(a_ones, n_boot=n_boot, func=np.median)
    assert_array_equal(out2, np.ones(n_boot))


def test_bootstrap_length():
    """Test that we get a bootstrap array of the right shape."""
    out = stat.bootstrap(a_norm)
    assert_equal(len(out), 10000)

    n_boot = 100
    out = stat.bootstrap(a_norm, n_boot=n_boot)
    assert_equal(len(out), n_boot)


def test_bootstrap_range():
    """Test that boostrapping a random array stays within the right range."""
    min, max = a_norm.min(), a_norm.max()
    out = stat.bootstrap(a_norm)
    nose.tools.assert_less_equal(min, out.min())
    nose.tools.assert_greater_equal(max, out.max())


def test_bootstrap_multiarg():
    """Test that bootstrap works with multiple input arrays."""
    x = np.vstack([[1, 10] for i in range(10)])
    y = np.vstack([[5, 5] for i in range(10)])

    test_func = lambda x, y: np.vstack((x, y)).max(axis=0)
    out_actual = stat.bootstrap(x, y, n_boot=2, func=test_func)
    out_wanted = np.array([[5, 10], [5, 10]])
    assert_array_equal(out_actual, out_wanted)


@raises(ValueError)
def test_bootstrap_arglength():
    """Test that different length args raise ValueError."""
    stat.bootstrap(range(5), range(10))


@raises(TypeError)
def test_bootstrap_noncallable():
    """Test that we get a TypeError with noncallable statfunc."""
    non_func = "mean"
    stat.bootstrap(a_norm, 100, non_func)


def test_percentiles():
    """Test function to return sequence of percentiles."""
    single_val = 5
    single = stat.percentiles(a_range, single_val)
    assert_equal(single, single_val)

    multi_val = [10, 20]
    multi = stat.percentiles(a_range, multi_val)
    assert_array_equal(multi, multi_val)

    array_val = np.random.randint(0, 101, 5).astype(float)
    array = stat.percentiles(a_range, array_val)
    assert_array_almost_equal(array, array_val)


def test_percentiles_acc():
    """Test accuracy of calculation."""
    # First a basic case
    data = np.array([10, 20, 30])
    val = 20
    perc = stat.percentiles(data, 50)
    assert_equal(perc, val)

    # Now test against scoreatpercentile
    percentiles = np.random.randint(0, 101, 10)
    out = stat.percentiles(a_norm, percentiles)
    for score, pct in zip(out, percentiles):
        assert_equal(score, sp.stats.scoreatpercentile(a_norm, pct))


def test_percentiles_axis():
    """Test use of axis argument to percentils."""
    data = np.random.randn(10, 10)

    # Test against the median with 50th percentile
    median1 = np.median(data)
    out1 = stat.percentiles(data, 50)
    assert_array_almost_equal(median1, out1)

    for axis in range(2):
        median2 = np.median(data, axis=axis)
        out2 = stat.percentiles(data, 50, axis=axis)
        assert_array_almost_equal(median2, out2)

    median3 = np.median(data, axis=0)
    out3 = stat.percentiles(data, [50, 95], axis=0)
    assert_array_almost_equal(median3, out3[0])
    assert_equal(2, len(out3))


def test_add_constant():
    """Test the add_constant function."""
    a = np.random.randn(10, 5)
    wanted = np.column_stack((a, np.ones(10)))
    got = stat.add_constant(a)
    assert_array_equal(wanted, got)


def test_highpass_matrix_shape():
    """Test the filter matrix is the right shape."""
    for n_tp in 10, 100:
        F = stat.fsl_highpass_matrix(n_tp, 50)
        assert_equal(F.shape, (n_tp, n_tp))


def test_filter_matrix_diagonal():
    """Test that the filter matrix has strong diagonal."""
    F = stat.fsl_highpass_matrix(10, 3)
    assert_array_equal(F.argmax(axis=1).squeeze(), np.arange(10))


def test_filtered_data_shape():
    """Test that filtering data returns same shape."""
    data = np.random.randn(100)
    data_filt = stat.fsl_highpass_filter(data, 30)
    assert_equal(data.shape, data_filt.shape)

    data = np.random.randn(100, 3)
    data_filt = stat.fsl_highpass_filter(data, 30)
    assert_equal(data.shape, data_filt.shape)


def test_filter_psd():
    """Test highpass filter with power spectral density."""
    a = np.array([np.sin(x) for x in np.linspace(0, 4 * np.pi, 100)])
    b = np.random.randn(100) / 2
    y = a + b
    y_filt = stat.fsl_highpass_filter(y, 10)
    assert_equal(y.shape, y_filt.shape)

    orig_psd, _ = psd(y, 2 ** 5)
    filt_psd, _ = psd(y_filt, 2 ** 5)

    nose.tools.assert_greater(orig_psd[:3].mean(), filt_psd[:3].mean())


def test_filter_strength():
    """Test that lower cutoff makes filter more aggresive."""
    a = np.sin(np.linspace(0, 4 * np.pi, 100))
    b = np.random.randn(100) / 2
    y = a + b

    cutoffs = np.linspace(20, 80, 5)
    densities = np.zeros_like(cutoffs)
    for i, cutoff in enumerate(cutoffs):
        filt = stat.fsl_highpass_filter(y, cutoff)
        density, _ = psd(filt, 2 ** 5)
        densities[i] = density.mean()

    assert_array_equal(densities, np.sort(densities))
