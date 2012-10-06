import os.path as op
import numpy as np
import nibabel as nib
import seaborn as sns

def plot_mask_distribution(fname, hist=False):
    """Plot the distribution of voxel coordinates in a mask file.

    Parameters
    ----------
    fname : string
        path to binary mask file

    Returns
    -------
    ax : matplotlib axis object
        axis with plot on it

    """
    img = nib.load(fname)
    mask = img.get_data()
    aff = img.get_affine()
    vox = np.where(mask)
    vox = np.vstack([vox, np.ones(mask.sum())])
    coords = np.dot(aff, vox)[:-1]
    colors = sns.get_color_list()[:3]
    for axis, data, color in zip(["x", "y", "z"], coords, colors):
        if hist:
            ax = sns.kdeplot(data, hist=True, label=axis, color=color)
        else:
            ax = sns.kdeplot(data, shade=True, label=axis, color=color)
    ax.legend()
    ax.set_title(op.basename(fname))
    return ax
