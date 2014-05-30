from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from six import string_types

from .nipy import VolumeImg


class Slicer(object):

    def __init__(self, anat, stat=None, mask=None, n_col=10, step=2,
                 tight=True):
        """Plot a mosiac of axial slices through an MRI volume.

        Parameters
        ----------
        anat : filename, nibabel image, or array
            The anatomical image that will form the background of the mosiac.
            If only an array is passed, an identity matrix will be used as
            the affine and orientation could be incorrect.
        stat : filename, nibabel image, or array
            A statistical map to plot as an overlay (which happens by calling
            one of the methods). If only an array is passed, it is assumed
            to have the same orientation as the anatomy.
        mask : filename, nibabel image, or array
            A binary image where voxels included in the statistical analysis
            are True. This will be used to gray-out voxels in the anatomical
            image that are outside the field of view. If you want to overlay
            the mask itself, pass it to ``stat``.
        n_col : int
            Number of columns in the mosaic.
        step : int
            Take every ``step`` slices along the z axis when plotting.
        tight_z : bool
            If True, only show slices that have voxels inside the mask.

        """
        # Load and reorient the anatomical image
        if isinstance(anat, string_types):
            anat_img = nib.load(anat)
            have_orientation = True
        elif isinstance(anat, np.ndarray):
            anat_img = nib.Nifti1Image(anat, np.eye(4))
            have_orientation = False
        else:
            anat_img = anat
            have_orientation = True
        self.anat_img = VolumeImg(anat_img.get_data(),
                                  anat_img.get_affine(),
                                  "mni").xyz_ordered(resample=True)
        self.anat_data = self.anat_img.get_data()
        anat_fov = self.anat_img.get_data() > 1e-5

        # Load and reorient the statistical image
        if stat is not None:
            if isinstance(stat, string_types):
                stat_img = nib.load(stat)
            elif isinstance(stat, np.ndarray):
                stat_img = nib.Nifti1Image(stat, anat_img.get_affine())
            else:
                stat_img = stat
            stat_data = np.nan_to_num(stat_img.get_data().astype(np.float))
            self.stat_img = VolumeImg(stat_data,
                                      stat_img.get_affine(),
                                      "mni").xyz_ordered(resample=True)

        # Load and reorient the mask image
        if mask is not None:
            if isinstance(mask, string_types):
                mask_img = nib.load(mask)
            elif isinstance(mask, np.ndarray):
                mask_img = nib.Nifti1Image(mask, anat_img.get_affine())
            else:
                mask_img = mask
            self.mask_img = VolumeImg(mask_img.get_data().astype(bool),
                                      mask_img.get_affine(),
                                      world_space="mn",
                                      interpolation="nearest",
                                      ).xyz_ordered(resample=True)
            mask_data = self.mask_img.get_data()
        else:
            mask_data = None

        # Find a field of view that tries to eliminate empty voxels
        if mask is None or not tight:
            self.fov = fov = anat_fov
            self.fov_slices = fov_slices = fov.any(axis=(0, 1))
        else:
            self.fov = fov = anat_fov | mask_data
            self.fov_slices = fov_slices = mask_data.any(axis=(0, 1))

        # Save the mosiac parameters
        self.n_col = n_col
        self.step = step

        # Sort out the slicers to focus on the brain and step properly
        self.start = start = np.argwhere(fov_slices).min()
        self.z_slice = slice(start, None, step)
        mask_x = np.argwhere(fov.any(axis=(1, 2)))
        self.x_slice = slice(mask_x.min() - 1, mask_x.max() + 1)
        mask_y = np.argwhere(fov.any(axis=(0, 2)))
        self.y_slice = slice(mask_y.min() - 1, mask_y.max() + 1)

        # Initialize the figure and plot the contant info
        self._setup_figure()
        self._plot_anat()
        if mask is not None:
            self._plot_inverse_mask()

        # Label the anatomy
        if have_orientation:
            self.fig.text(.01, .03, "L", size=14, color="w",
                          ha="left", va="center")
            self.fig.text(.99, .03, "R", size=14, color="w",
                          ha="right", va="center")

    def _setup_figure(self):
        """Initialize the figure and axes."""
        n_slices = self.fov_slices.sum() // self.step

        n_row = np.ceil(n_slices / self.n_col)
        nx, ny, _ = self.anat_data[self.x_slice, self.y_slice].shape
        figsize = self.n_col, (ny / nx) * n_row
        plot_kws = dict(nrows=int(n_row), ncols=int(self.n_col),
                        figsize=figsize, facecolor="k")

        self.fig, self.axes = plt.subplots(**plot_kws)
        [ax.set_axis_off() for ax in self.axes.flat]
        self.fig.subplots_adjust(0, 0, 1, 1, 0, 0)

    def _plot_anat(self):
        """Plot the anatomy in grayscale."""
        anat_data = self.anat_img.get_data()
        vmin, vmax = 0, anat_data[self.fov].max() * 1.1
        anat_fov = anat_data[self.x_slice, self.y_slice, self.z_slice]
        anat_slices = anat_fov.transpose(2, 0, 1)
        for data, ax in zip(anat_slices, self.axes.flat):
            ax.imshow(np.rot90(data), cmap="Greys_r", vmin=vmin, vmax=vmax)

        empty_slices = len(self.axes.flat) - len(anat_slices)
        if empty_slices > 0:
            shape = np.rot90(data).shape
            for ax in self.axes.flat[-empty_slices:]:
                ax.imshow(np.zeros(shape), cmap="Greys_r", vmin=0, vmax=10)

    def _plot_inverse_mask(self):
        """Dim the voxels outside of the statistical analysis FOV."""
        mask_data = self.mask_img.get_data().astype(np.bool)
        anat_data = self.anat_img.get_data()
        mask_data = np.where(mask_data | (anat_data < 1e-5), np.nan, 1)
        mask_fov = mask_data[self.x_slice, self.y_slice, self.z_slice]
        mask_slices = mask_fov.transpose(2, 0, 1)
        for data, ax in zip(mask_slices, self.axes.flat):
            ax.imshow(np.rot90(data), cmap="bone", interpolation="nearest",
                      alpha=.5, vmin=0, vmax=3)

    def plot_activation(self, thresh=2, vmin=None, vmax=None, vmax_perc=99,
                        pos_cmap="Reds_r", neg_cmap=None, alpha=1):
        """Plot the stat image as uni- or bi-polar activation with a threshold.

        Parameters
        ----------
        thresh : float
            Threshold value for the statistic; overlay will not be visible
            between -thresh and thresh.
        vmin, vmax : floats
            The anchor values for the colormap. The same values will be used
            for the positive and negative overlay.
        vmax_perc : int
            The percentile of the data (within the fov and above the threshold)
            at which to saturate the colormap by default. Overriden if a there
            is a specific value passed for vmax.
        pos_cmap, neg_cmap : names of colormaps or colormap objects
            The colormapping for the positive and negative overlays.
        alpha : float
            The transparancy of the overlay.
        
        """
        stat_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        pos_data = stat_data.copy()
        pos_data[pos_data < thresh] = np.nan
        if vmin is None:
            vmin = thresh
        if vmax is None:
            calc_data = stat_data[np.abs(stat_data) > thresh]
            vmax = np.percentile(np.abs(calc_data), vmax_perc)

        pos_slices = pos_data.transpose(2, 0, 1)
        for data, ax in zip(pos_slices, self.axes.flat):
            ax.imshow(np.rot90(data), cmap=pos_cmap,
                      vmin=vmin, vmax=vmax, alpha=alpha)

        if neg_cmap is not None:
            thresh, nvmin, nvmax = -thresh, -vmax, -vmin
            neg_data = stat_data.copy()
            neg_data[neg_data > thresh] = np.nan
            neg_slices = neg_data.transpose(2, 0, 1)
            for data, ax in zip(neg_slices, self.axes.flat):
                ax.imshow(np.rot90(data), cmap=neg_cmap,
                          vmin=nvmin, vmax=nvmax, alpha=alpha)

            self._add_double_colorbar(vmin, vmax, pos_cmap, neg_cmap)
        else:
            self._add_single_colorbar(vmin, vmax, pos_cmap)

    def plot_overlay(self, cmap, vmin=None, vmax=None,
                     vmin_perc=1, vmax_perc=99, thresh=None, alpha=1):
        """Plot the stat image as a single overlay with a threshold.

        Parameters
        ----------
        cmap : name of colormap or colormap object
            The colormapping for the overlay.
        vmin, vmax : floats
            The anchor values for the colormap. The same values will be used
            for the positive and negative overlay.
        vmin_perc, vmax_perc : ints
            The percentiles of the data (within the fov and above the threshold)
            that will be anchor points for the colormap by default. Overriden if
            specific values are passed for vmin or vmax.
        thresh : float
            Threshold value for the statistic; overlay will not be visible
            between -thresh and thresh.
        pos_cmap, neg_cmap : names of colormaps or colormap objects
            The colormapping for the positive and negative overlays.
        alpha : float
            The transparancy of the overlay.
        
        """
        stat_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        if hasattr(self, "mask_img"):
            fov = self.mask_img.get_data()[self.x_slice,
                                           self.y_slice,
                                           self.z_slice]
        else:
            fov = np.ones_like(stat_data).astype(bool)

        if vmin is None:
            vmin = np.percentile(stat_data[fov], vmin_perc)
        if vmax is None:
            vmax = np.percentile(stat_data[fov], vmax_perc)
        if thresh is not None:
            stat_data[stat_data < thresh] = np.nan

        stat_data[~fov] = np.nan

        slices = stat_data.transpose(2, 0, 1)
        for data, ax in zip(slices, self.axes.flat):
            ax.imshow(np.rot90(data), cmap=cmap,
                      vmin=vmin, vmax=vmax, alpha=alpha)

        self._add_single_colorbar(vmin, vmax, cmap)

    def _pad_for_cbar(self):
        """Add extra space to the bottom of the figure for the colorbars."""
        w, h = self.fig.get_size_inches()
        cbar_inches = .3
        self.fig.set_size_inches(w, h + cbar_inches)
        cbar_height = cbar_inches / (h + cbar_inches)
        self.fig.subplots_adjust(0, cbar_height, 1, 1)

        #  Needed so things look nice in the notebook
        bg_ax = self.fig.add_axes([0, 0, 1, cbar_height])
        bg_ax.set_axis_off()
        bg_ax.pcolormesh(np.array([[1]]), cmap="Greys", vmin=0, vmax=1)

        return cbar_height

    def _add_single_colorbar(self, vmin, vmax, cmap):
        """Add colorbars for a single overlay."""
        cbar_height = self._pad_for_cbar()
        cbar_ax = self.fig.add_axes([.3, .01, .4, cbar_height - .01])
        cbar_ax.set(xticks=[], yticks=[])
        for side, spine in cbar_ax.spines.items():
            spine.set_visible(False)

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        cbar_ax.pcolormesh(bar_data, cmap=cmap)

        self.fig.text(.29, .005 + cbar_height * .5, "%.2g" % vmin,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.71, .005 + cbar_height * .5, "%.2g" % vmax,
                      color="white", size=14, ha="left", va="center")

    def _add_double_colorbar(self, vmin, vmax, pos_cmap, neg_cmap):
        """Add colorbars for a positive and a negative overlay."""
        cbar_height = self._pad_for_cbar()

        pos_ax = self.fig.add_axes([.55, .01, .3, cbar_height - .01])
        pos_ax.set(xticks=[], yticks=[])
        for side, spine in pos_ax.spines.items():
            spine.set_visible(False)

        neg_ax = self.fig.add_axes([.15, .01, .3, cbar_height - .01])
        neg_ax.set(xticks=[], yticks=[])
        for side, spine in neg_ax.spines.items():
            spine.set_visible(False)

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        pos_ax.pcolormesh(bar_data, cmap=pos_cmap)
        neg_ax.pcolormesh(bar_data, cmap=neg_cmap)

        self.fig.text(.54, .005 + cbar_height * .5, "%.2g" % vmin,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.86, .005 + cbar_height * .5, "%.2g" % vmax,
                      color="white", size=14, ha="left", va="center")

        self.fig.text(.14, .005 + cbar_height * .5, "%.2g" % -vmax,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.46, .005 + cbar_height * .5, "%.2g" % -vmin,
                      color="white", size=14, ha="left", va="center")
