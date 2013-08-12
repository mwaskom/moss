import os
import re
import os.path as op

import numpy as np
import pandas as pd
import nibabel as nib


def locate_peaks(vox_coords):
    """Find most probable region in HarvardOxford Atlas of a vox coord."""
    sub_names = harvard_oxford_sub_names
    ctx_names = harvard_oxford_ctx_names
    at_dir = op.join(os.environ["FSLDIR"], "data", "atlases")
    ctx_data = nib.load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-cort-prob-2mm.nii.gz")).get_data()
    sub_data = nib.load(op.join(at_dir, "HarvardOxford",
                            "HarvardOxford-sub-prob-2mm.nii.gz")).get_data()

    loc_list = []
    for coord in vox_coords:
        coord = tuple(coord)
        ctx_index = np.argmax(ctx_data[coord])
        ctx_prob = ctx_data[coord][ctx_index]
        sub_index = np.argmax(sub_data[coord])
        sub_prob = sub_data[coord][sub_index]

        if not max(sub_prob, ctx_prob):
            loc_list.append(("Unknown", 0))
            continue
        if not ctx_prob and sub_index in [0, 11]:
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        if sub_prob > ctx_prob and sub_index not in [0, 1, 11, 12]:
            loc_list.append((sub_names[sub_index], sub_prob))
            continue
        loc_list.append((ctx_names[ctx_index], ctx_prob))

    return pd.DataFrame(loc_list, columns=["MaxProb Region", "Prob"])


def shorten_name(region_name, atlas):
    """Implement regexp sub for verbose Harvard Oxford Atlas region."""
    sub_list = dict(ctx=harvard_oxford_ctx_subs,
                    sub=harvard_oxford_sub_subs)
    for pat, rep in sub_list[atlas]:
        region_name = re.sub(pat, rep, region_name).strip()
    return region_name


def vox_to_mni(vox_coords):
    """Given ijk voxel coordinates, return xyz from image affine."""
    try:
        fsldir = os.environ["FSLDIR"]
    except KeyError:
        raise RuntimeError("vox_to_mni requires FSLDIR to be defined.")
    mni_file = op.join(fsldir, "data/standard/avg152T1.nii.gz")
    aff = nib.load(mni_file).get_affine()
    mni_coords = np.zeros_like(vox_coords)
    for i, coord in enumerate(vox_coords):
        coord = coord.astype(float)
        mni_coords[i] = np.dot(aff, np.r_[coord, 1])[:3].astype(int)
    return mni_coords


harvard_oxford_sub_subs = [
    ("Left", "L"),
    ("Right", "R"),
    ("Cerebral Cortex", "Ctx"),
    ("Cerebral White Matter", "Cereb WM"),
    ("Lateral Ventrica*le*", "LatVent"),
]

harvard_oxford_ctx_subs = [
    ("Superior", "Sup"),
    ("Middle", "Mid"),
    ("Inferior", "Inf"),
    ("Lateral", "Lat"),
    ("Medial", "Med"),
    ("Frontal", "Front"),
    ("Parietal", "Par"),
    ("Temporal", "Temp"),
    ("Occipital", "Occ"),
    ("Cingulate", "Cing"),
    ("Cortex", "Ctx"),
    ("Gyrus", "G"),
    ("Sup Front G", "SFG"),
    ("Mid Front G", "MFG"),
    ("Inf Front G", "IFG"),
    ("Sup Temp G", "STG"),
    ("Mid Temp G", "MTG"),
    ("Inf Temp G", "ITG"),
    ("Parahippocampal", "Parahip"),
    ("Juxtapositional", "Juxt"),
    ("Intracalcarine", "Intracalc"),
    ("Supramarginal", "Supramarg"),
    ("Supracalcarine", "Supracalc"),
    ("Paracingulate", "Paracing"),
    ("Fusiform", "Fus"),
    ("Orbital", "Orb"),
    ("Opercul[ua][mr]", "Oper"),
    ("temporooccipital", "tempocc"),
    ("triangularis", "triang"),
    ("opercularis", "oper"),
    ("division", ""),
    ("par[st] *", ""),
    ("anterior", "ant"),
    ("posterior", "post"),
    ("superior", "sup"),
    ("inferior", "inf"),
    (" +", " "),
    ("\(.+\)", ""),
]

harvard_oxford_sub_names = [
    'L Cereb WM',
    'L Ctx',
    'L LatVent',
    'L Thalamus',
    'L Caudate',
    'L Putamen',
    'L Pallidum',
    'Brain-Stem',
    'L Hippocampus',
    'L Amygdala',
    'L Accumbens',
    'R Cereb WM',
    'R Ctx',
    'R LatVent',
    'R Thalamus',
    'R Caudate',
    'R Putamen',
    'R Pallidum',
    'R Hippocampus',
    'R Amygdala',
    'R Accumbens']

harvard_oxford_ctx_names = [
    'Front Pole',
    'Insular Ctx',
    'SFG',
    'MFG',
    'IFG, triang',
    'IFG, oper',
    'Precentral G',
    'Temp Pole',
    'STG, ant',
    'STG, post',
    'MTG, ant',
    'MTG, post',
    'MTG, tempocc',
    'ITG, ant',
    'ITG, post',
    'ITG, tempocc',
    'Postcentral G',
    'Sup Par Lobule',
    'Supramarg G, ant',
    'Supramarg G, post',
    'Angular G',
    'Lat Occ Ctx, sup',
    'Lat Occ Ctx, inf',
    'Intracalc Ctx',
    'Front Med Ctx',
    'Juxt Lobule Ctx',
    'Subcallosal Ctx',
    'Paracing G',
    'Cing G, ant',
    'Cing G, post',
    'Precuneous Ctx',
    'Cuneal Ctx',
    'Front Orb Ctx',
    'Parahip G, ant',
    'Parahip G, post',
    'Lingual G',
    'Temp Fus Ctx, ant',
    'Temp Fus Ctx, post',
    'Temp Occ Fus Ctx',
    'Occ Fus G',
    'Front Oper Ctx',
    'Central Oper Ctx',
    'Par Oper Ctx',
    'Planum Polare',
    'Heschl"s G',
    'Planum Tempe',
    'Supracalc Ctx',
    'Occ Pole']
