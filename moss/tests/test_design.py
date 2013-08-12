from __future__ import division
import numpy as np
import nose.tools as nt
import numpy.testing as npt
from .. import design


def test_cb1_ideal():

    # Test basic balanced
    evs = [10, 10]
    ideal = np.array([[.5, .5], [.5, .5]])
    npt.assert_array_equal(ideal, design.cb1_ideal(evs))

    # Test basic unbalanced
    evs = [10, 20]
    ideal = np.array([[1 / 3, 2 / 3], [1 / 3, 2 / 3]])
    npt.assert_array_equal(ideal, design.cb1_ideal(evs))

    # Ensure matrix size
    evs = range(5)
    nt.assert_equal((5, 5), design.cb1_ideal(evs).shape)


def test_cb1_prob():

    # Test basic
    sched = [0, 1, 0, 1]
    evs = [2, 2]
    expected = np.array([[0, .5], [1, 0]])
    npt.assert_array_equal(expected, design.cb1_prob(sched, evs))


def test_cb1_cost():

    # Test perfect transition matrix
    # (this won't actually ever happen)
    ideal = np.array([[.5, .5], [.5, .5]])
    test = ideal
    npt.assert_equal(0.0, design.cb1_cost(ideal, test))

    # Now something less rosy
    test = np.array([[.4, .6], [.6, .4]])
    nt.assert_almost_equal(0.2, design.cb1_cost(ideal, test))
