from nose.tools import assert_equal

from .. import misc


def test_sig_stars():

    assert_equal(misc.sig_stars(.5), "")
    assert_equal(misc.sig_stars(.07), ".")
    assert_equal(misc.sig_stars(.03), "*")
    assert_equal(misc.sig_stars(0.004), "**")
    assert_equal(misc.sig_stars(0.00001), "***")
