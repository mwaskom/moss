from __future__ import division
import numpy as np
import pandas as pd
import nose.tools as nt
import numpy.testing as npt
import pandas.util.testing as pdt
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


class TestEVCreation(object):

    data = pd.DataFrame(dict(cue=[6, 12, 18, 24],
                             stim=["face", "house", "face", "house"],
                             rt=[1.5, 1.2, 1.8, 1.5],
                             run=[1, 1, 1, 1]))

    def test_simple_ev(self):

        ev = design.build_simple_ev(self.data, "cue", "event")

        npt.assert_array_equal(ev.columns, ["run", "onset", "value",
                                            "condition", "duration"])

        npt.assert_array_equal(ev.onset, self.data.cue)
        npt.assert_array_equal(ev.value, np.ones(4))
        npt.assert_array_equal(ev.condition, ["event"] * 4)
        npt.assert_array_equal(ev.duration, np.zeros(4))

    def test_condition_ev(self):

        ev = design.build_condition_ev(self.data, "cue", "stim")

        npt.assert_array_equal(ev.columns, ["run", "onset", "condition",
                                            "value", "duration"])

        npt.assert_array_equal(ev.onset, self.data.cue)
        npt.assert_array_equal(ev.value, np.ones(4))
        npt.assert_array_equal(ev.condition, self.data.stim)
        npt.assert_array_equal(ev.duration, np.zeros(4))

        ev = design.build_condition_ev(self.data, "cue", "stim", prefix="a_")
        npt.assert_array_equal(ev.condition, "a_" + self.data.stim)

    def test_parametric_ev(self):

        ev = design.build_parametric_ev(self.data, "cue", "resp_time", "rt")

        npt.assert_array_equal(ev.columns, ["run", "onset", "value",
                                            "condition", "duration"])

        npt.assert_array_equal(ev.onset, self.data.cue)
        npt.assert_array_almost_equal(ev.value, [0, -.3, .3, 0])
        npt.assert_array_equal(ev.condition, ["resp_time"] * 4)
        npt.assert_array_equal(ev.duration, np.zeros(4))

        ev = design.build_parametric_ev(self.data, "cue", "resp_time", "rt",
                                        center=1)
        npt.assert_array_almost_equal(ev.value, [.5, .2, .8, .5])

        scale = lambda x: x / (x.abs().max())

        ev = design.build_parametric_ev(self.data, "cue", "resp_time", "rt",
                                        scale=scale)
        npt.assert_array_almost_equal(ev.value, [0, -1, 1, 0])

    def test_duration(self):

        ev = self.data[["cue", "run"]].copy()
        ev.columns = ["onset", "run"]

        # Test impulse
        ev = design._add_duration_information(self.data, ev, None)
        npt.assert_array_equal(ev["duration"], np.zeros(4))

        # Test constant
        ev = design._add_duration_information(self.data, ev, 1)
        npt.assert_array_equal(ev["duration"], np.ones(4))

        # Test from column
        ev = design._add_duration_information(self.data, ev, "rt")
        npt.assert_array_equal(ev["duration"], self.data["rt"])
