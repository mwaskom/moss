import numpy as np
import scipy

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
    out1 = stat.bootstrap(a_ones, n_boot)
    assert_array_equal(out1, np.ones(n_boot))
    out2 = stat.bootstrap(a_ones, n_boot, np.median)
    assert_array_equal(out2, np.ones(n_boot))


def test_bootstrap_length():
    """Test that we get a bootstrap array of the right shape."""
    out = stat.bootstrap(a_norm)
    assert_equal(len(out), 10000)

    n_boot = 100
    out = stat.bootstrap(a_norm, n_boot)
    assert_equal(len(out), n_boot)


def test_bootstrap_range():
    """Test that boostrapping a random array stays within the right range."""
    min, max = a_norm.min(), a_norm.max()
    out = stat.bootstrap(a_norm)
    nose.tools.assert_less_equal(min, out.min())
    nose.tools.assert_greater_equal(max, out.max())


@raises(TypeError)
def test_bootstrap_noncallable():
    """Test that we get a TypeError with noncallable statfunc."""
    non_func = "mean"
    stat.bootstrap(a_norm, 100, non_func)


@raises(ValueError)
def test_bootstrap_indim():
    """Test that we get a ValueError with multidimensional input."""
    a_bad = np.random.rand(10, 10)
    stat.bootstrap(a_bad)


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
        assert_equal(score, scipy.stats.scoreatpercentile(a_norm, pct))


def test_pmf_hist_basics():
    """Test the function to return barplot args for pmf hist."""
    out = stat.pmf_hist(a_norm)
    assert_equal(len(out), 3)
    x, h, w = out
    assert_equal(len(x), len(h))

    # Test simple case
    a = np.arange(10)
    x, h, w = stat.pmf_hist(a, 10)
    nose.tools.assert_true(np.all(h == h[0]))


def test_pmf_hist_widths():
    """Test histogram width is correct."""
    x, h, w = stat.pmf_hist(a_norm)
    assert_equal(x[1] - x[0], w)


def test_pmf_hist_normalization():
    """Test that output data behaves like a PMF."""
    x, h, w = stat.pmf_hist(a_norm)
    nose.tools.assert_almost_equal(sum(h), 1)
    nose.tools.assert_less_equal(h.max(), 1)


def test_pmf_hist_bins():
    """Test bin specification."""
    x, h, w = stat.pmf_hist(a_norm, 20)
    assert_equal(len(x), 20)
