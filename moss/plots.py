import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns


def plot_mask_distribution(mask_img, hist=False, ax=None):
    """Plot the distribution of voxel coordinates in a mask image or file.

    Parameters
    ----------
    fname : string or nibabel image
        path to binary mask file or image object with data and affine
    hist : bool, optional
        if True plots histogram instead of shading density plot
    ax : matplotlib axis, optional
        plots on this axis, if given

    Returns
    -------
    ax : matplotlib axis object
        axis with plot on it

    """
    if ax is None:
        ax = plt.subplot(111)
    if isinstance(mask_img, basestring):
        img = nib.load(mask_img)
    else:
        img = mask_img
    mask = img.get_data()
    aff = img.get_affine()
    vox = np.where(mask)
    vox = np.vstack([vox, np.ones(mask.sum())])
    coords = np.dot(aff, vox)[:-1]
    colors = sns.get_color_list()[:3]
    for axis, data, color in zip(["x", "y", "z"], coords, colors):
        if hist:
            sns.kdeplot(data, hist=True, label=axis, color=color, ax=ax)
        else:
            sns.kdeplot(data, shade=True, label=axis, color=color, ax=ax)
    ax.legend()
    return ax
