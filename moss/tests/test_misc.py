import numpy as np
from nose.tools import assert_equal

from .. import misc


def test_sig_stars():
    """Test p-value decorations, even though p-values are dumb."""
    assert_equal(misc.sig_stars(.5), "")
    assert_equal(misc.sig_stars(.07), ".")
    assert_equal(misc.sig_stars(.03), "*")
    assert_equal(misc.sig_stars(0.004), "**")
    assert_equal(misc.sig_stars(0.00001), "***")


def test_iqr():
    """Test the IQR function."""
    a = np.arange(5)
    iqr = misc.iqr(a)
    assert_equal(iqr, 2)
