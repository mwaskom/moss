from __future__ import division
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import nose
import nose.tools as nt
import numpy.testing as npt
from .. import slicer

has_fsl = "FSLDIR" in os.environ


class TestSlicer(object):

    if not has_fsl:
        raise nose.SkipTest

    mni_file = os.path.join(os.environ["FSLDIR"],
                            "data/standard/avg152T1_brain.nii.gz")
    mni_img = nib.load(mni_file)
    mni_data = mni_img.get_data()

    def test_slicer_init_path(self):

        slicer.Slicer(self.mni_file)
        plt.close("all")

    def test_slicer_init_img(self):

        slicer.Slicer(self.mni_img)
        plt.close("all")

    def test_slicer_init_array(self):

        slicer.Slicer(self.mni_data)
        plt.close("all")

    def test_mosiac_cols(self):

        slc = slicer.Slicer(self.mni_img, n_col=5)
        nt.assert_equal(slc.axes.shape[1], 5)
        plt.close("all")

    def test_mosiac_step(self):

        step1 = slicer.Slicer(self.mni_img, step=1)
        step2 = slicer.Slicer(self.mni_img, step=2)
        nt.assert_equal(len(step2.axes.flat) * 2, len(step1.axes.flat))
        plt.close("all")

    def test_moasic_tight(self):

        mask = (self.mni_data * 0).astype(np.int8)
        mask[:, :, 40:60] = 1
        slc = slicer.Slicer(self.mni_img, mask=mask, step=1, tight=True)
        nt.assert_equal(len(slc.axes.flat), 20)
        plt.close("all")

    def test_anat_img(self):

        slc = slicer.Slicer(self.mni_img, step=1)
        plot_data = slc.anat_img.get_data()
        want_image = np.rot90(plot_data[slc.x_slice,
                                        slc.y_slice,
                                        slc.z_slice][:, :, 10])
        got_image = slc.axes.flat[10].images[0].get_array()
        npt.assert_array_equal(want_image, got_image)
        plt.close("all")
