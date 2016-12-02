"""Miscellaneous utility functions."""
import gzip
import itertools
import numpy as np
import pandas as pd
from scipy import stats

import six.moves.cPickle as pickle


def df_to_struct(df):
    """Converts a DataFrame to RPy-compatible structured array."""
    struct_array = df.to_records()
    arr_dtype = struct_array.dtype.descr
    for i, dtype in enumerate(arr_dtype):
        if dtype[1] == np.dtype('object'):
            arr_dtype[i] = (dtype[0], dtype[1].replace("|O", "|S"))

    struct_array = np.asarray([tuple(d) for d in struct_array],
                              dtype=arr_dtype)

    return struct_array


def df_ttest(df, by, key, paired=False, nice=True, **kwargs):
    """Perform a T-test over a DataFrame groupby."""
    test_kind = "rel" if paired else "ind"
    test_func = getattr(stats, "ttest_" + test_kind)
    args = [d[key] for i, d in df.groupby(by)]
    t, p = test_func(*args, **kwargs)
    dof = (len(df) / 2) - 1 if paired else len(df) - 2
    if nice:
        return "t(%d) = %.3f; p = %.3g%s" % (dof, t, p, sig_stars(p))
    else:
        return pd.Series([t, p], ["t", "p"])


def df_oneway(df, by, key, nice=True, **kwargs):
    """Perform a oneway analysis over variance on a DataFrame groupby."""
    args = [d[key] for i, d in df.groupby(by)]
    f, p = stats.f_oneway(*args, **kwargs)
    dof_b = len(args) - 1
    dof_w = len(df) - dof_b
    if nice:
        return "F(%d, %d) = %.3f; p = %.3g%s" % (dof_b, dof_w, f,
                                                 p, sig_stars(p))
    else:
        return pd.Series([f, p], ["F", "p"])


def product_index(values, names=None):
    """Make a MultiIndex from the combinatorial product of the values."""
    iterable = itertools.product(*values)
    idx = pd.MultiIndex.from_tuples(list(iterable), names=names)
    return idx


def make_master_schedule(evs):
    """Take a list of event specifications and make one schedule.

    Parameters
    ----------
    evs : sequence of n x 3 arrays
        list of (onset, duration, amplitude) event secifications

    Returns
    -------
    sched : n_event x 5 array
        schedule of event specifications with
        event and presentation ids

    """
    evs = np.asarray(evs)
    n_cond = len(evs)

    # Make a vector of condition ids and stimulus indices
    cond_ids = [np.ones(evs[i].shape[0]) * i for i in range(n_cond)]
    cond_ids = np.concatenate(cond_ids)
    stim_idxs = np.concatenate([np.arange(len(ev)) for ev in evs])

    # Make a schedule of the whole run
    sched = np.row_stack(evs)
    sched = np.column_stack((sched, cond_ids, stim_idxs))

    # Sort the master schedule by onset time
    timesorter = np.argsort(sched[:, 0])
    sched = sched[timesorter]
    return sched


def sig_stars(p):
    """Return a R-style significance string corresponding to p values."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "."
    return ""


def iqr(a):
    """Calculate the IQR for an array of numbers."""
    a = np.asarray(a)
    q1 = stats.scoreatpercentile(a, 25)
    q3 = stats.scoreatpercentile(a, 75)
    return q3 - q1


class Results(object):
    """Extremely simple namespace for passing around and pickling data."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def load_pkl(fname, zip=True):
    """Read pickled data from disk, possible decompressing."""
    if zip:
        open = gzip.open
    with open(fname, "rb") as fid:
        res = pickle.load(fid)
    return res


def save_pkl(fname, res, zip=True):
    """Write pickled data to disk, possible compressing."""
    if zip:
        open = gzip.open
    with open(fname, "wb") as fid:
        pickle.dump(res, fid)
