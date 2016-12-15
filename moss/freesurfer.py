from __future__ import division

import os
import os.path as op

import numpy as np
from scipy.spatial import KDTree

import nibabel as nib
import nibabel.freesurfer as nifs
from nibabel.affines import apply_affine


def vol_to_surf_xfm(vol_fname, reg_fname):
    """Obtain a transformation from vol voxels -> Freesurfer surf coords.
    Parameters
    ----------
    vol_fname : string
        Filename pointing at image defining the vol space.
    reg_fname : string
        Filename pointing at registration file (from bbregister) that maps
        ``vol_fname`` to the Freesurfer anatomy.

    Returns
    -------
    xfm : 4 x 4 numpy array
        Transformation matrix that can be applied to surf coords.

    """
    # Load the Freesurfer "tkreg" style transform file
    # Confusingly, this file actually encodes the anat-to-func transform
    anat2func_xfm = np.genfromtxt(reg_fname, skip_header=4, skip_footer=1)
    func2anat_xfm = np.linalg.inv(anat2func_xfm)

    # Get a tkreg-compatibile mapping from IJK to RAS
    vol_img = nib.load(vol_fname)
    mgh_img = nib.MGHImage(np.zeros(vol_img.shape[:3]),
                           vol_img.affine,
                           vol_img.header)
    vox2ras_tkr = mgh_img.header.get_vox2ras_tkr()

    # Combine the two transformations
    xfm = np.dot(func2anat_xfm, vox2ras_tkr)

    return xfm


def surf_to_voxel_coords(subj, hemi, xfm, surf="graymid",
                         subjects_dir=None):
    """Obtain voxel coordinates of surface vertices in the EPI volume.

    Parameters
    ----------
    subj : string
        Freesurfer subject ID.
    hemi : lh | rh
        Hemisphere of surface to map.
    xfm : 4 x 4 array
        Linear transformation matrix between spaces.
    surf : string
        Freesurfer surface name defining coords.

    Returns
    -------
    i, j, k : 1d int arrays
        Arrays of voxel indices.

    """
    # Load the surface geometry
    if subjects_dir is None:
        subjects_dir = os.environ["SUBJECTS_DIR"]
    surf_fname = os.path.join(subjects_dir, subj, "surf",
                              "{}.{}".format(hemi, surf))
    coords, _ = nib.freesurfer.read_geometry(surf_fname)

    return apply_affine(xfm, coords).round().astype(np.int).T



