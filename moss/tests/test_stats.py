import numpy as np

from numpy.testing import assert_array_equal
import nose.tools
from nose.tools import assert_equal, raises

from .. import stats


a_norm = np.random.randn(100)


def test_bootstrap():
    """Test that bootstrapping gives the right answer in dumb cases."""
    a_ones = np.ones(10)
    n_boot = 5
    out1 = stats.bootstrap(a_ones, n_boot)
    assert_array_equal(out1, np.ones(n_boot))
    out2 = stats.bootstrap(a_ones, n_boot, np.median)
    assert_array_equal(out2, np.ones(n_boot))


def test_bootstrap_length():
    """Test that we get a bootstrap array of the right shape."""
    out = stats.bootstrap(a_norm)
    assert_equal(len(out), 10000)

    n_boot = 100
    out = stats.bootstrap(a_norm, n_boot)
    assert_equal(len(out), n_boot)


def test_bootstrap_range():
    """Test that boostrapping a random array stays within the right range."""
    min, max = a_norm.min(), a_norm.max()
    out = stats.bootstrap(a_norm)
    nose.tools.assert_less_equal(min, out.min())
    nose.tools.assert_greater_equal(max, out.max())


@raises(TypeError)
def test_bootstrap_noncallable():
    """Test that we get a TypeError with noncallable statfunc."""
    non_func = "mean"
    stats.bootstrap(a_norm, 100, non_func)


@raises(ValueError)
def test_bootstrap_indim():
    """Test that we get a ValueError with multidimensional input."""
    a_bad = np.random.rand(10, 10)
    stats.bootstrap(a_bad)
