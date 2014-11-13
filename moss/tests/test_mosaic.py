from __future__ import division
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

import nose
import nose.tools as nt
import numpy.testing as npt
from .. import mosaic


class TestMosaic(object):

    if "FSLDIR" not in os.environ:
        raise nose.SkipTest

    anat_file = os.path.join(os.environ["FSLDIR"],
                             "data/standard/avg152T1_brain.nii.gz")
    anat_img = nib.load(anat_file)
    anat_data = anat_img.get_data()

    mask_file = os.path.join(os.environ["FSLDIR"],
                             "data/standard/MNI152_T1_2mm_brain_mask.nii.gz")
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_data()

    rs = np.random.RandomState(99)
    stat_data = mask_data.astype(np.float) * rs.normal(0, 1, mask_data.shape)
    stat_img = nib.Nifti1Image(stat_data, mask_img.get_affine())

    def test_mosaic_init_path(self):

        mosaic.Mosaic(self.anat_file)
        plt.close("all")

    def test_mosaic_init_img(self):

        mosaic.Mosaic(self.anat_img)
        plt.close("all")

    def test_mosaic_init_array(self):

        mosaic.Mosaic(self.anat_data)
        plt.close("all")

    def test_mosaic_cols(self):

        slc = mosaic.Mosaic(self.anat_img, n_col=5)
        nt.assert_equal(slc.axes.shape[1], 5)
        plt.close("all")

    def test_mosaic_step(self):

        step1 = mosaic.Mosaic(self.anat_img, n_col=10, step=1)
        step2 = mosaic.Mosaic(self.anat_img, n_col=10, step=2)
        nt.assert_equal(len(step2.axes.flat) * 2, len(step1.axes.flat))
        plt.close("all")

    def test_moasic_tight(self):

        mask = (self.anat_data * 0).astype(np.int8)
        mask[:, :, 40:67] = 1
        slc = mosaic.Mosaic(self.anat_img, mask=mask, step=1, tight=True)
        nt.assert_equal(len(slc.axes.flat), 27)
        plt.close("all")

    def test_mosaic_full_anat(self):

        m1 = mosaic.Mosaic(self.anat_img, tight=False)

        full_img = nib.Nifti1Image(np.ones_like(self.anat_data),
                                   self.anat_img.get_affine())
        m2 = mosaic.Mosaic(full_img)
        nt.assert_equal(m1.axes.shape, m2.axes.shape)
        nt.assert_equal(m2.axes.flat[0].images[0].get_array().T.shape,
                        self.anat_data.shape[:-1])

    def test_anat_image_data(self):

        slc = mosaic.Mosaic(self.anat_img)
        plot_data = slc.anat_img.get_data()
        want_image = np.rot90(plot_data[slc.x_slice,
                                        slc.y_slice,
                                        slc.z_slice][:, :, 10])
        got_image = slc.axes.flat[10].images[0].get_array()
        npt.assert_array_equal(want_image, got_image)
        plt.close("all")

    def test_mask_image_data(self):

        slc = mosaic.Mosaic(self.anat_img, mask=self.mask_img)
        mask_data = slc.mask_img.get_data()
        mask_sliced = np.rot90(mask_data[slc.x_slice,
                                         slc.y_slice,
                                         slc.z_slice][:, :, 10])
        mask_image = slc.axes.flat[10].images[1].get_array()
        mask_overlap = mask_image * mask_sliced
        nt.assert_true(not mask_overlap.any())
        plt.close("all")

    def test_statistical_overlays(self):

        slc1 = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc2 = mosaic.Mosaic(self.anat_img, self.stat_img)

        slc1.plot_activation(thresh=1, vmin=.5, vmax=1.5,
                             pos_cmap="Purples_r", alpha=.9)
        slc2.plot_overlay(thresh=1, vmin=.5, vmax=1.5,
                          cmap="Purples_r", alpha=.9)

        for ax1, ax2 in zip(slc1.axes.flat, slc2.axes.flat):
            npt.assert_array_almost_equal(ax1.images[1].get_array().data,
                                          ax2.images[1].get_array().data)
        plt.close("all")

    def test_subthresh_statistical_overlay(self):

        slc = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc.plot_activation(thresh=100)

        for ax in slc.axes.flat:
            assert np.isnan(ax.images[1].get_array().data).all()

        plt.close("all")

    def test_bipolar_overlays(self):

        slc = mosaic.Mosaic(self.anat_img, self.stat_img)

        slc.plot_activation(thresh=1, vmin=.5, vmax=1.5,
                            neg_cmap="Blues",  alpha=.9)

        ax = slc.axes.flat[10]
        nt.assert_equal(len(ax.images), 3)
        pos = ax.images[1].get_array()
        neg = ax.images[2].get_array()
        nt.assert_true(pos[~np.isnan(pos)].min() >= 1)
        nt.assert_true(neg[~np.isnan(neg)].max() <= -1)
        plt.close("all")

    def test_statistical_overlay_by_map(self):

        slc1 = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc2 = mosaic.Mosaic(self.anat_img)

        slc1.plot_overlay(vmin=-1, vmax=1, cmap="coolwarm", alpha=.9)
        slc2.map("imshow", self.stat_img, vmin=-1, vmax=1,
                 cmap="coolwarm", alpha=.9)

        for ax1, ax2 in zip(slc1.axes.flat, slc2.axes.flat):
            npt.assert_array_almost_equal(ax1.images[1].get_array().data,
                                          ax2.images[1].get_array().data)
        plt.close("all")

    def test_mask_overlay(self):

        slc = mosaic.Mosaic(self.anat_img, self.mask_img)
        slc.plot_mask()
        overlay_data = slc.axes.flat[10].images[1].get_array().data
        overlay_vals = np.unique(np.nan_to_num(overlay_data))
        npt.assert_array_equal(overlay_vals, [0, 1])
        plt.close("all")

    def test_empty_overlay(self):

        slc = mosaic.Mosaic(self.anat_img, np.zeros_like(self.stat_data))
        slc.plot_activation(2)
        slc.plot_overlay("coolwarm")

    def test_cubehelix_overlay(self):

        slc = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc.plot_overlay("cube:0:.5", 0)
        slc.plot_overlay("cube:1.5:-1.5", 0)
        slc.plot_overlay("cube:2:-1_r", 0)

    def test_colormap_text(self):

        slc = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc.plot_overlay("Purples", 0, 10.5, fmt="%.1f")
        nt.assert_equal(slc.fig.texts[-1].get_text(), "10.5")

        slc = mosaic.Mosaic(self.anat_img, self.stat_img)
        slc.plot_overlay("Purples", 0, 10.5, fmt="%d")
        nt.assert_equal(slc.fig.texts[-1].get_text(), "10")
