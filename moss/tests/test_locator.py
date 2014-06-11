import os
import numpy as np
import pandas as pd
from nose.plugins.skip import SkipTest
from numpy.testing import assert_equal

from .. import locator


def test_locate_peaks():
    """Test that known peaks are located properly."""
    if "FSLDIR" not in os.environ:
        raise SkipTest

    challenge = [
        ([(60, 60, 50)], ("L Cereb WM", 100)),
        ([(62, 69, 50)], ("MFG", 20)),
        ([(31, 50, 27)], ("Parahip G, post", 30)),
        ([(26, 55, 27)], ("Temp Fus Ctx, post", 3)),
        ([(29, 50, 30)], ("R Hippocampus", 95))]

    for coord, res in challenge:
        res = dict(zip(["MaxProb Region", "Prob"], list(res)))
        res = pd.DataFrame(res, index=[0])
        out = np.array(locator.locate_peaks(coord))
        out[0, -1] = int(out[0, -1])
        yield assert_equal, np.array(res), out


def test_shorten_name():
    """Test that verbose Harvard Oxford ROI names are shorted."""
    names = [("Parahippocampal Gyrus, anterior division",
              "Parahip G, ant",
              "ctx"),
             ("Middle Frontal Gyrus", "MFG", "ctx"),
             ("Right Hippocampus", "R Hippocampus", "sub")]

    for orig, new, atlas in names:
        yield assert_equal, new, locator.shorten_name(orig, atlas)


def test_vox_to_mni():
    """Test the voxel index to MNI coordinate transformation."""
    if "FSLDIR" not in os.environ:
        raise SkipTest

    coords = [((29, 68, 57), (32, 10, 42)),
              ((70, 38, 42), (-50, -50, 12)),
              ((45, 63, 36), (0, 0, 0))]

    for vox, mni in coords:
        vox = np.atleast_2d(vox)
        mni = np.atleast_2d(mni)
        yield assert_equal, mni, locator.vox_to_mni(vox)
