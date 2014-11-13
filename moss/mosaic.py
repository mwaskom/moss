from __future__ import division
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nib
from six import string_types

from .nipy import VolumeImg


class Mosaic(object):

    def __init__(self, anat=None, stat=None, mask=None, n_col=9, step=2,
                 tight=True, show_mask=True, stat_interp="continuous"):
        """Plot a mosaic of axial slices through an MRI volume.

        Parameters
        ----------
        anat : filename, nibabel image, or array
            The anatomical image that will form the background of the mosaic.
            If only an array is passed, an identity matrix will be used as
            the affine and orientation could be incorrect. If absent, try
            to find the FSL data and uses the MNI152 brain.
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
        show_mask : bool
            If True, gray-out voxels in the anat image that are outside
            of the mask image.
        stat_interp : continuous | nearest
            The kind of interpolation to perform (if necessary) when
            reorienting the statistical image.

        """
        # Load and reorient the anatomical image
        if anat is None:
            if "FSLDIR" in os.environ:
                anat = os.path.join(os.environ["FSLDIR"],
                                    "data/standard/avg152T1_brain.nii.gz")
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
                                  world_space="mni",
                                  ).xyz_ordered(resample=True)
        self.anat_data = self.anat_img.get_data()

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
                                      world_space="mni",
                                      interpolation=stat_interp,
                                      ).xyz_ordered(resample=True)

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
                                      world_space="mni",
                                      interpolation="nearest",
                                      ).xyz_ordered(resample=True)
            mask_data = self.mask_img.get_data()
        else:
            mask_data = None

        # Re-mask the anat and stat images
        # This is only really useful when the image affines had rotations
        # and thus we needed to interpolate
        if mask_data is not None:
            if stat is not None:
                self.stat_img._data[~mask_data] = 0

            # Use an implicit/explicit mask to zero out very dim voxels
            # This helps deal with the interpolation error for non-MNI
            # data which otherwise causes a weird ring around the brain
            thresh = np.percentile(self.anat_data[mask_data], 98) * .05
            self.anat_img._data[self.anat_data < thresh] = 0

        # Find a field of view with nonzero anat voxels
        anat_fov = self.anat_img.get_data() > 1e-5

        # Find a field of view that tries to eliminate empty voxels
        if mask is None or not tight:
            self.fov = fov = anat_fov
            self.fov_slices = fov_slices = fov.any(axis=(0, 1))
        else:
            self.fov = fov = anat_fov | mask_data
            self.fov_slices = fov_slices = mask_data.any(axis=(0, 1))

        # Save the mosaic parameters
        self.n_col = n_col
        self.step = step

        # Sort out the mosaics to focus on the brain and step properly
        self.start = start = np.argwhere(fov_slices).min()
        self.z_slice = slice(start, None, step)
        mask_x = np.argwhere(fov.any(axis=(1, 2)))
        self.x_slice = slice(max(mask_x.min() - 1, 0), mask_x.max() + 1)
        mask_y = np.argwhere(fov.any(axis=(0, 2)))
        self.y_slice = slice(max(mask_y.min() - 1, 0), mask_y.max() + 1)

        # Initialize the figure and plot the contant info
        self._setup_figure()
        self._plot_anat()
        if mask is not None and show_mask:
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
        vmin, vmax = 0, anat_data[self.fov].max()
        anat_fov = anat_data[self.x_slice, self.y_slice, self.z_slice]
        self._map("imshow", anat_fov, cmap="Greys_r", vmin=vmin, vmax=vmax)

        empty_slices = len(self.axes.flat) - anat_fov.shape[2]
        if empty_slices > 0:
            i, j, _ = anat_fov.shape
            for ax in self.axes.flat[-empty_slices:]:
                ax.imshow(np.zeros((i, j)), cmap="Greys_r", vmin=0, vmax=10)

    def _plot_inverse_mask(self):
        """Dim the voxels outside of the statistical analysis FOV."""
        mask_data = self.mask_img.get_data().astype(np.bool)
        anat_data = self.anat_img.get_data()
        mask_data = np.where(mask_data | (anat_data < 1e-5), np.nan, 1)
        mask_fov = mask_data[self.x_slice, self.y_slice, self.z_slice]
        self._map("imshow", mask_fov, cmap="bone", vmin=0, vmax=3,
                  interpolation="nearest", alpha=.5)

    def _map(self, func_name, data, **kwargs):
        """Apply a named function to a 3D volume of data on each axes."""
        slices = data.transpose(2, 0, 1)
        for slice, ax in zip(slices, self.axes.flat):
            func = getattr(ax, func_name)
            func(np.rot90(slice), **kwargs)

    def plot_activation(self, thresh=2, vmin=None, vmax=None, vmax_perc=99,
                        vfloor=None, pos_cmap="Reds_r", neg_cmap=None,
                        alpha=1, fmt="%.2g"):
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
        vfloor : float or None
            If not None, this sets the vmax value, if the value at the provided
            vmax_perc does not exceed it.
        pos_cmap, neg_cmap : names of colormaps or colormap objects
            The colormapping for the positive and negative overlays.
        alpha : float
            The transparancy of the overlay.
        fmt : %-style format string
            Format of the colormap annotation

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
            if calc_data.any():
                vmax = np.percentile(np.abs(calc_data), vmax_perc)
            else:
                vmax = vmin * 2

        pos_cmap = self._get_cmap(pos_cmap)

        self._map("imshow", pos_data, cmap=pos_cmap,
                  vmin=vmin, vmax=vmax, alpha=alpha)

        if neg_cmap is not None:
            thresh, nvmin, nvmax = -thresh, -vmax, -vmin
            neg_data = stat_data.copy()
            neg_data[neg_data > thresh] = np.nan

            neg_cmap = self._get_cmap(neg_cmap)

            self._map("imshow", neg_data, cmap=neg_cmap,
                      vmin=nvmin, vmax=nvmax, alpha=alpha)

            self._add_double_colorbar(vmin, vmax, pos_cmap, neg_cmap, fmt)
        else:
            self._add_single_colorbar(vmin, vmax, pos_cmap, fmt)

    def plot_overlay(self, cmap, vmin=None, vmax=None, center=False,
                     vmin_perc=1, vmax_perc=99, thresh=None,
                     alpha=1, fmt="%.2g", colorbar=True):
        """Plot the stat image as a single overlay with a threshold.

        Parameters
        ----------
        cmap : name of colormap or colormap object
            The colormapping for the overlay.
        vmin, vmax : floats
            The anchor values for the colormap. The same values will be used
            for the positive and negative overlay.
        center : bool
            If true, center the colormap. This respects the larger absolute
            value from the other (vmin, vmax) arguments, but overrides the
            smaller one.
        vmin_perc, vmax_perc : ints
            The percentiles of the data (within fov and above threshold)
            that will be anchor points for the colormap by default. Overriden
            if specific values are passed for vmin or vmax.
        thresh : float
            Threshold value for the statistic; overlay will not be visible
            between -thresh and thresh.
        alpha : float
            The transparancy of the overlay.
        fmt : %-style format string
            Format of the colormap annotation
        colorbar : bool
            If true, add a colorbar.

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
            if stat_data.any():
                vmax = np.percentile(stat_data[fov], vmax_perc)
            else:
                vmax = vmin * 2
        if center:
            vabs = max(np.abs(vmin), vmax)
            vmin, vmax = -vabs, vabs
        if thresh is not None:
            stat_data[stat_data < thresh] = np.nan

        stat_data[~fov] = np.nan

        cmap = self._get_cmap(cmap)

        self._map("imshow", stat_data, cmap=cmap,
                  vmin=vmin, vmax=vmax, alpha=alpha)

        if colorbar:
            self._add_single_colorbar(vmin, vmax, cmap, fmt)

    def plot_mask(self, color="#3cb371", alpha=.66):
        """Plot the statistical volume as a binary mask."""
        mask_data = self.stat_img.get_data()[self.x_slice,
                                             self.y_slice,
                                             self.z_slice]
        bool_mask = mask_data.astype(bool)
        mask_data = bool_mask.astype(np.float)
        mask_data[~bool_mask] = np.nan

        cmap = mpl.colors.ListedColormap([color])
        self._map("imshow", mask_data, cmap=cmap, vmin=.5, vmax=1.5,
                  interpolation="nearest", alpha=alpha)

    def plot_contours(self, cmap, levels=8, linewidths=1):
        """Plot the statistical volume as a contour map."""
        slices = self.stat_img.get_data()[self.x_slice,
                                          self.y_slice,
                                          self.z_slice].transpose(2, 0, 1)

        if isinstance(cmap, list):
            levels = len(cmap)
            cmap = mpl.colors.ListedColormap(cmap)

        vmin, vmax = np.percentile(slices, [1, 99])
        for slice, ax in zip(slices, self.axes.flat):
            try:
                ax.contour(np.rot90(slice), levels, cmap=cmap,
                           vmin=vmin, vmax=vmax, linewidths=linewidths)
            except ValueError:
                pass

    def plot_mask_edges(self, palette="husl", linewidths=.75):
        """Plot the edges of possibly multiple masks to show overlap."""
        from seaborn import color_palette
        n_masks = self.stat_img._data.shape[-1]
        cmap = mpl.colors.ListedColormap(color_palette(palette, n_masks))

        for mask_num in range(n_masks):
            slices = self.stat_img.get_data()[self.x_slice,
                                              self.y_slice,
                                              self.z_slice,
                                              mask_num].transpose(2, 0, 1)

            for slice, ax in zip(slices, self.axes.flat):
                if slice.any():
                    ax.contour(np.rot90(slice * (mask_num + 1)), n_masks,
                               cmap=cmap, vmin=1, vmax=n_masks,
                               linewidths=linewidths)

        self._add_single_colorbar(1, n_masks, cmap, "%d")

    def map(self, func_name, data, thresh=None, **kwargs):
        """Map a dataset across the mosaic of axes.

        Parameters
        ----------
        func_name : str
            Name of a pyplot function.
        data : filename, nibabel image, or array
            Dataset to plot.
        thresh : float
            Don't map voxels in ``data`` below this threshold.
        kwargs : key, value mappings
            Other keyword arguments are passed to the plotting function.

        """
        if isinstance(data, string_types):
            data_img = nib.load(data)
        elif isinstance(data, np.ndarray):
            data_img = nib.Nifti1Image(data, np.eye(4))
        else:
            data_img = data
        data = VolumeImg(data_img.get_data(), data_img.get_affine(),
                         "mni").xyz_ordered(resample=True).get_data()
        data = data.astype(np.float)
        data[data < thresh] = np.nan
        data = data[self.x_slice, self.y_slice, self.z_slice]
        self._map(func_name, data, **kwargs)

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

    def _add_single_colorbar(self, vmin, vmax, cmap, fmt):
        """Add colorbars for a single overlay."""
        cbar_height = self._pad_for_cbar()
        cbar_ax = self.fig.add_axes([.3, .01, .4, cbar_height - .01])
        cbar_ax.set(xticks=[], yticks=[])
        for side, spine in cbar_ax.spines.items():
            spine.set_visible(False)

        bar_data = np.linspace(0, 1, 256).reshape(1, 256)
        cbar_ax.pcolormesh(bar_data, cmap=cmap)

        self.fig.text(.29, .005 + cbar_height * .5, fmt % vmin,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.71, .005 + cbar_height * .5, fmt % vmax,
                      color="white", size=14, ha="left", va="center")

    def _add_double_colorbar(self, vmin, vmax, pos_cmap, neg_cmap, fmt):
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

        self.fig.text(.54, .005 + cbar_height * .5, fmt % vmin,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.86, .005 + cbar_height * .5, fmt % vmax,
                      color="white", size=14, ha="left", va="center")

        self.fig.text(.14, .005 + cbar_height * .5, fmt % -vmax,
                      color="white", size=14, ha="right", va="center")
        self.fig.text(.46, .005 + cbar_height * .5, fmt % -vmin,
                      color="white", size=14, ha="left", va="center")

    def _get_cmap(self, cmap):
        """Parse a string spec of a cubehelix palette."""
        from seaborn import cubehelix_palette
        if isinstance(cmap, string_types):
            if cmap.startswith("cube"):
                if cmap.endswith("_r"):
                    reverse = False
                    cmap = cmap[:-2]
                else:
                    reverse = True
                _, start, rot = cmap.split(":")
                cmap = cubehelix_palette(as_cmap=True,
                                         start=float(start),
                                         rot=float(rot),
                                         light=.95,
                                         dark=0,
                                         reverse=reverse)
        return cmap

    def savefig(self, fname, **kwargs):
        """Save the figure."""
        self.fig.savefig(fname, facecolor="k", edgecolor="k", **kwargs)

    def close(self):
        """Close the figure."""
        plt.close(self.fig)
